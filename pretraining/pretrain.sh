#!/bin/bash

if [ "$#" -ne 3 ]; then
    echo "Illegal number of parameters"
fi


basepath="~/bp/pretraining"
wd="$SCRATCH/albert"
model_name="csbase1"
echo Creating work directory in $wd

trainbase=`basename $1`
testbase=`basename $2`

mkdir $wd
cp -r $basepath/ALBERT $basepath/configs $wd/
cp -r $basepath/models/$model_name $wd/model
mkdir $wd/dataset
cp $1* $wd/dataset/
cp $2* $wd/dataset/
cp $basepath/*.py $wd

cd $wd
export MODEL_DIR="configs/$model_name"
export TRAIN_DATA="dataset/$trainbase"
export TEST_DATA="dataset/$testbase"
# export PROCESSED_DATA="datasets/tenten256/test1.pre"
export PRETRAINED_MODEL="model/"
mkdir $PRETRAINED_MODEL

echo Running screens

screen -mdS bash bash

screen -mdS tensorboard \
singularity exec --nv \
    -B /storage/praha1/home/zep:/storage/praha1/home/zep \
    -B $SCRATCH:$SCRATCH \
    ~/tensor.simg \
    bash -c "sleep 30 && tensorboard --logdir $PRETRAINED_MODEL"

screen -md -S albert \
singularity exec --nv \
    -B /storage/praha1/home/zep:/storage/praha1/home/zep \
    -B $SCRATCH:$SCRATCH \
    ~/tensor.simg \
    python3 autobert.py \
        configs/base.yaml \
        $TRAIN_DATA \
        $TEST_DATA \
        $PRETRAINED_MODEL \
        $3 \
        250

echo Waiting for screen to finish
while screen -list | grep -q albert
do
    sleep 10
done

echo Saving new model

rm -r dataset/
rm -r $basepath/models_bak/$model_name
mv $basepath/models/$model_name $basepath/models_bak/$model_name
cp -r $PRETRAINED_MODEL $basepath/models/$model_name

