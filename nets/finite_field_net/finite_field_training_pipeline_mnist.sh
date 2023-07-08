#!/bin/bash -l

#SBATCH --nodes=1 # Allocate *at least* 1 node to this job.
#SBATCH --ntasks=1 # Allocate *at most* 1 task for job steps in the job
#SBATCH --cpus-per-task=1 # Each task needs only one CPU
#SBATCH --mem=16G # This particular job won't need much memory
#SBATCH --time=2-00:00:00  # 2 day
#SBATCH --mail-user=ubasa001@ucr.edu
#SBATCH --mail-type=ALL
#SBATCH --job-name="MNIST Finite Field Domain Training"
#SBATCH -p cpu # You could pick other partitions for other jobs
#SBATCH --wait-all-nodes=1  # Run once all resources are available
#SBATCH --output=int_mnist_output_%j-%N.txt # logging per job and per host in the current directory. Both stdout and stderr are logged.

conda activate ff-net
python finite_field_training_pipeline_mnist.py
