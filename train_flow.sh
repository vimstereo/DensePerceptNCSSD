#!/bin/bash
mkdir -p checkpoints
python -u train_flow.py --name raft-kitti  --stage sceneflow --validation kitti --gpus 0 1 --num_steps 100000 --batch_size 4 --lr 0.0001 --image_size 288 960 --wdecay 0.00001 --gamma=0.85
