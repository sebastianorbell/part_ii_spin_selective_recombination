#!/bin/bash
#
#$ -S /bin/bash

# Set some variables.
#
SCR=$TMPDIR
ORIG=`pwd`

# Set input file name here - must have a '.in' suffix
INFILE=

# Create output file name
OUTFILE=`basename $INFILE .in`.out

module load openmpi2

# Set correct path
export PATH=$ORCA:$PATH

# Copy input file to scratch dir
cp $INFILE  $SCR

cd $SCR

# Finally, run job
$ORCA/orca  $INFILE > $OUTFILE 2> $ERRFILE

# Copy files back to original directory
cp * $ORIG

cd $ORIG
