SAVE_DIR='.'  # 你的混合数据集路径
DATA_NAME='mixed_music_speech_100'  # 你的数据集名称
python vocab.py \
--dataset_name ${DATA_NAME} \
--save_dir ${SAVE_DIR}