#!/bin/bash
#
#$ -S /bin/bash
#$ -cwd
#$ -M sebastian.orbell@sjoh.ox.ac.uk
#$ -l s_rt=120:00:00
#$ -pe smp 4

# Set some variables.
#
SCR=$TMPDIR
ORIG=`pwd`

# Copy input file to scratch dir
cp * $SCR

cd $SCR

source /home/dirac/oxford/sjoh4236/anaconda3/bin/activate
export OMP_NUM_THREADS=$NSLOTS
# Finally, run job
python3 fn1_dirac_313.py > log.out > ERRFILE

# Copy files back to original directory
cp * $ORIG

cd $ORIG
