#!/bin/bash -l
#SBATCH --job-name=test_serial
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=2G
#SBATCH --time=00:20:00
#SBATCH --mail-user=rene.huertgen@stud.uni-hannover.de
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output test_serial-job_%j.out
#SBATCH --error test_serial-job_%j.err
#SBATCH --gpus=1
 
# Change to my work dir
# SLURM_SUBMIT_DIR is an environment variable that automatically gets
# assigned the directory from which you did submit the job. A batch job
# is like a new login, so you'll initially be in your HOME directory. 
# So it's usually a good idea to first change into the directory you did
# submit your job from.
cd $SLURM_SUBMIT_DIR
 
 
# Load the modules you need, see corresponding page in the cluster documentation
module load Miniconda3
module load GCC/11.2.0
module load CUDA/12.3.0

conda activate gaussian_splatting_rl
export PYTHONPATH=$PYTHONPATH:/bigwork/nhmlhuer/git/gaussian_splatting_rl/src/ 
# Start my serial app
# srun is needed here only to create an entry in the accounting system,
# but you could also start your app without it here, since it's only serial.
srun python full_eval.py
