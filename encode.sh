export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9
cd ./data
JSON_PATH='/tsdata2/dhy/data/mixed_dataset/mixed_music_speech_100/mixed_dataset.json' # change to your path
SAVE_DIR='./mixed_music_speech_100' # change to your path
ENCODEC_PATH='../../../models/SSR-Speech-English/wmencodec.th' # change to your wmencodec path
DATA_NAME='mixed_music_speech_100' # change to yours
python encode.py \
--dataset_name ${DATA_NAME} \
--save_dir ${SAVE_DIR} \
--encodec_model_path ${ENCODEC_PATH} \
--json_path ${JSON_PATH} \
--start 0 \
--end 10000000