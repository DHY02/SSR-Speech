JSON_PATH='/tsdata2/dhy/data/mixed_dataset/mixed_music_speech_100/mixed_dataset.json' # change to your path
SAVE_DIR='./' # change to your path
DATA_NAME='mixed_music_speech_100' # change to yours
SAVENAME='mixed_music_speech_100' # change to yours
# python filemaker.py \
# --dataset_name ${DATA_NAME} \
# --json_path ${JSON_PATH} \
# --save_dir ${SAVE_DIR} \
# --savename ${SAVENAME}

python filemaker_split.py \
    --json_path ${JSON_PATH} \
    --dataset_name ${DATA_NAME} \
    --save_dir ${SAVE_DIR} \
    --train_ratio 0.9 \
    --seed 42