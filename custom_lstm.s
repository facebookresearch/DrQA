#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=6:00:00
#SBATCH --mem=64GB
#SBATCH --job-name=DrQA_train
#SBATCH --mail-type=END
#SBATCH --mail-user=jp.park@nyu.edu
#SBATCH --output=output/train_%j.out
#SBATCH --gres=gpu:1
  
module purge
module load cuda/9.0.176
module load cudnn/9.0v7.0.5 
#module load pytorch/python3.6/0.3.0_4
#module load python3/intel/3.6.3  
module load gcc/6.3.0
RUNDIR=$SCRATCH/my_project/run-${SLURM_JOB_ID/.*}
mkdir -p $RUNDIR
SRCDIR=$SCRATCH/DrQA/scripts/reader/
cd $RUNDIR
export CLASSPATH=$CLASSPATH:$SCRATCH/data/corenlp/*
$HOME/anaconda3/bin/python $SRCDIR/train.py --model-dir=$SCRATCH/DrQA/models --rnn-type='custom_lstm' --concat-rnn-layers=False --bidirectional=False


