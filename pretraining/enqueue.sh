#!/bin/bash

# Enqueue pretraining jobs for ALBERT on MetaCentrum

dataset=1           # first part of dataset
end=10              # last part of dataset

maxstep=30000       # last training step
stepstep=20000      # training steps per datset part

cmdstart="qsub -q gpu -l select=ngpus=2:cluster=adan:ncpus=4:mem=24gb:scratch_local=30gb"
depend="-W depend=afterok:"

model="cslarge3"                    # model
bpath="bakalarka/pretraining"       # repo path
dpath="$bpath/datasets/tta512"      # dataset path
fname="tta_prep"                    # dataset part file name


get_cmd() {
# crete the command for queuing the pretraining
	script="$bpath/pretrain.sh $model $dpath/${fname}_1$1.txt.pre $dpath/ttt_prep.txt.pre $2"
	if [ "$#" -lt 3 ]; then
		cmd="$cmdstart -- $script"
	else
		cmd="$cmdstart $depend$3 -- $script"
	fi
}


# set the jobid which the first job should depend on (if any)
jobid=""
#jobid="1561637.meta-pbs.metacentrum.cz"


# enqueue all jobs
while [ $dataset -le $end ];
do
	n=$(printf %02d $dataset)
	get_cmd $n $maxstep $jobid
	echo $cmd
	jobid="`$cmd`"
	echo $jobid
	dataset=$(($dataset + 1))
	maxstep=$(($maxstep + $stepstep))
done
