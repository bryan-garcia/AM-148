#!/bin/bash
#SBATCH --job-name=ptest     # Job name
#SBATCH --output=ptest.log
#SBATCH --error=perror.log
#SBATCH --partition=am-148-s20
#SBATCH --qos=am-148-s20
#SBATCH --account=am-148-s20
#SBATCH --mail-type=END      # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=bgarci26@ucsc.edu   # Where to send mail
#SBATCH --nodes=1                    # Number of nodes

srun make lib
srun export SO_PATH=$(pwd)
srun python plink.py
