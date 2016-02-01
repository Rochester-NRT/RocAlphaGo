#!/bin/bash
#SBATCH -J parallelTest
#SBATCH -p gpu-debug
#SBATCH -t 00-00:01:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 1
#SBATCH --mem-per-cpu=4gb
#SBATCH -o /gpfs/fs1/home/ttrine/Documents/alphaGoReplication/parallelTest.out
#SBATCH --gres=gpu:1

module load torch
module load cuda/7.5

#run the scripts
luajit policyNet.lua
luajit valueNet.lua
