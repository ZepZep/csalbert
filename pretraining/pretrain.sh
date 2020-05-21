#!/bin/bash

# script to pretrain ALBERT model on MetaCentrum

if [ "$#" -ne 4 ]; then
    echo "Illegal number of parameters"
    echo usage pretrain [model_name] [train] [test] [endstep]
fi


basepath=~/bakalarka/pretraining
wd="$SCRATCH/albert"
model_name="$1"
trainbase=`basename $2`
testbase=`basename $3`


# ------------------------------------------------
# Create working directory on fast SSD in $SCRATCH
echo Creating work directory in $wd

mkdir $wd
mkdir $wd/albert
cp -r $basepath/albert/* $wd/albert
cp -r $basepath/configs $wd/
cp -r $basepath/models/$model_name $wd/model/
mkdir $wd/dataset
cp $2* $wd/dataset/
cp $3* $wd/dataset/
cp $basepath/*.py $wd

cd $wd
export TRAIN_DATA="dataset/$trainbase"
export TEST_DATA="dataset/$testbase"
export PRETRAINED_MODEL="model/"
mkdir $PRETRAINED_MODEL


# ------------------------------------------------
# run screen sessions for monitoring
# tensorflow programs ar started in singularity container
echo Running screens

# a bash screen for full access to the machine
screen -mdS bash bash

# tensorboard server screen to be able to view information about training
screen -mdS tensorboard \
singularity exec --nv \
    -B /storage/praha1/home/zep:/storage/praha1/home/zep \
    -B $SCRATCH:$SCRATCH \
    ~/tensor.simg \
    bash -c "sleep 30 && tensorboard --logdir $PRETRAINED_MODEL"

# autobert screen, the actual pretraining script
screen -md -L -S albert \
singularity exec --nv \
    -B /storage/praha1/home/zep:/storage/praha1/home/zep \
    -B $SCRATCH:$SCRATCH \
    ~/tensor.simg \
    python3 autobert.py \
        configs/$model_name.yaml \
        $TRAIN_DATA \
        $TEST_DATA \
        $PRETRAINED_MODEL \
        $4 \
        250

        
# ------------------------------------------------
# because the screens are detached, we need to check when they are finished
echo Waiting for screen to finish
while screen -list | grep -q albert
do
    sleep 10
done


# ------------------------------------------------
# Save the model and clean up the working directory
echo Saving new model

rm -r dataset/
rm -r $basepath/models_bak/$model_name
mv $basepath/models/$model_name $basepath/models_bak/$model_name
cp -r $PRETRAINED_MODEL $basepath/models/$model_name
