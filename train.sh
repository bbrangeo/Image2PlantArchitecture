#!/bin/bash

python src/train_2.py \
    --dataset_url "https://huggingface.co/datasets/heesup/Cowpea-Architecture-XML-WDS/resolve/main/shard-{000000..000200}.tar" \
    --encoder_checkpoint facebook/dinov2-small \
    --decoder_checkpoint gpt2-medium \
    --image_size 448 \
    --batch_size 4 \
    --epoch 4