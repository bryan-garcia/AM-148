#!/bin/bash
#SBATCH --job-name=mandelbrot      # Job name
#SBATCH --output=mandelbrot_test.log
#SBATCH --partition=am-148-s20
#SBATCH --mail-type=ALL         # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=bgarci26@ucsc.edu   # Where to send mail
#SBATCH --nodes=1                    # Number of nodes

srun make
srun ./mandelbrot
