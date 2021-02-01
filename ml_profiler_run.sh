#!/bin/sh

MODEL_LIST="alexnet efficientnet inception lenet mobilenet mobilenetv2 squeezenet xception zfnet"
BATCH_SIZE_LIST="32 64 128 256"
OPT_LIST="SGD Adam Adagrad Momentum"
LEARN_RATE_LIST="0.1 0.01 0.001 0.0001 0.00001"
EPOCH=20

for model in ${MODEL_LIST}
do
  for batch in ${BATCH_SIZE_LIST}
  do
    for opt in ${OPT_LIST}
    do
      for lr in ${LEARN_RATE_LIST}
      do
        rm -rf __pycache__
        python3 ml_profiler.py -m ${model} -b ${batch} -o ${opt} -r ${lr} -e ${EPOCH}
      done
    done
  done
done
