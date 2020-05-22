#!/bin/bash
#
#$ -S /bin/bash
#$ -cwd
#$ -l s_rt=120:00:00
#$ -pe smp 12

# Set some variables.
#
SCR=$TMPDIR
ORIG=`pwd`

# Copy input file to scratch dir
cp *  $SCR

cd $SCR


# Finally, run job
python3 ./fn1_dirac_303.py

# Copy files back to original directory
cp * $ORIG

cd $ORIG
