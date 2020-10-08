#!/bin/zsh

### TODO: fix file structure and script to automatically change config and checkpoint
### TODO: automate detection of all videos
#MODEL_CHECKPOINT="/home/filipkr/Documents/xjob/mmpose/checkpoints/top-down/hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918.pth"
MODEL_CHECKPOINT="/home/filipkr/Documents/xjob/mmpose/checkpoints/top-down/hrnet_w48_coco_384x288_dark-741844ba_20200812.pth"
#MODEL_CONFIG="/home/filipkr/Documents/xjob/mmpose/configs/top_down/darkpose/coco-wholebody/hrnet_w48_coco_wholebody_384x288_dark_plus.py"
MODEL_CONFIG="/home/filipkr/Documents/xjob/mmpose/configs/top_down/darkpose/coco/hrnet_w48_coco_384x288_dark.py"
VIDEO="/home/filipkr/Documents/xjob/vids/real/vids-before-after/002/Baseline/002FL.MTS"
OUT_DIR="/home/filipkr/Documents/xjob/vids/results-fl/002/"
FILE_NAME=""
ONLY_BOX=false

python demo/top_down_detection00.py $MODEL_CONFIG $MODEL_CHECKPOINT --video-path $VIDEO --out-video-root $OUT_DIR
