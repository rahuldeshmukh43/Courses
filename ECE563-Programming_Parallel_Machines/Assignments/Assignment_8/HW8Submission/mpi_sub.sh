#!/bin/bash
# FILENAME: mpi.sub
#SBATCH  --nodes=1
#SBATCH  --ntasks-per-node=16
#SBATCH  --time=00:01:00
#SBATCH  -A scholar
# module load intel; module load impi

srun --mpi=pmi2 -n 16 ./mm | tee mpi_out.txt

./seq | tee seq_out.txt

