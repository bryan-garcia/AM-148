#!/bin/bash
#SBATCH --job-name=gauss     # Job name
#SBATCH --output=gauss.log
#SBATCH --partition=am-148-s20
#SBATCH --qos.am-148-s20
#SBATCH --account=am-148-s20
#SBATCH --mail-type=END      # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=bgarci26@ucsc.edu   # Where to send mail
#SBATCH --nodes=1                    # Number of nodes

srun make
srun ./gauss
