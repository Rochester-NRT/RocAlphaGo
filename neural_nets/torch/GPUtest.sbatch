#!/bin/bash
#SBATCH -J parallelTest
#SBATCH -p gpu-debug
#SBATCH -t 00-00:01:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 1
#SBATCH --mem-per-cpu=4gb
#SBATCH -o /gpfs/fs1/home/ttrine/Documents/AlphaGoReplication/torch/GPUtest.out
#SBATCH --gres=gpu:1

module load torch
module load cuda/7.5

#each script below should print its model architecture
luajit p_pi.lua
luajit p_sigma.lua
luajit v_theta.lua
