#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1
export WORLD_SIZE=2

dataset=mixed_music_speech_100
mkdir -p ./logs/${dataset}

exp_root="../exp_results"
exp_name=e830M_ssrspeech_finetune  # 修改实验名称以区分微调
dataset_dir="/root/autodl-tmp/data/mixed_music_speech_100"
encodec_codes_folder_name="wmencodec"
load_model_path="../pretrained_models/SSR-Speech-English/English.pth"

export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 
torchrun --nnodes=1 --rdzv-backend=c10d --rdzv-endpoint=localhost:41977 --nproc_per_node=${WORLD_SIZE} \
../main.py \
--seed 42 \
--precision "float16" \
--num_workers 32 \
--resume \
--tb_write_every_n_steps 50 \
--print_every_n_steps 200 \
--val_every_n_steps 1500 \
--finetune_mode \
--lr 0.0001 \
--max_num_tokens 3000 \
--val_max_num_tokens 3000 \
--num_buckets 6 \
--dynamic_batching 1 \
--weight_decay 0.01 \
--warmup_fraction 0.05 \
--num_steps 30000 \
--gradient_accumulation_steps 16 \
--gradient_clip_val 0.5 \
--early_stop_step 3000 \
--early_stop_threshold 0.01 \
--optimizer_name "AdamW" \
--pad_x 0 \
--audio_max_length 15 \
--audio_min_length 1 \
--text_max_length 400 \
--text_min_length 5 \
--encodec_sr 50 \
--drop_long 1 \
--mask_len_min 1 \
--mask_len_max 600 \
--eos 2051 \
--tts_enhanced 1 \
--cfg_enhanced 0 \
--predict_mask_token 1 \
--predict_all 0 \
--reduced_eog 0 \
--special_first 0 \
--n_special 5 \
--codebook_weight "[5,1,0.5,0.1]" \
--max_mask_portion 0.7 \
--max_n_spans 3 \
--shuffle_mask_embedding 0 \
--mask_sample_dist "uniform" \
--min_gap 5 \
--n_codebooks 4 \
--text_vocab_size 100 \
--text_pad_token 100 \
--audio_vocab_size "2048" \
--empty_token 2048 \
--eog 2049 \
--audio_pad_token 2050 \
--sos 2052 \
--mts 2053 \
--d_model 2048 \
--audio_embedding_dim 2048 \
--text_embedding_dropout 0.1 \
--audio_embedding_dropout 0.1 \
--text_positional_embedding_dropout 0.1 \
--audio_positional_embedding_dropout 0.1 \
--trm_dropout 0.15 \
--nhead 16 \
--num_decoder_layers 16 \
--phn_folder_name "phonemes" \
--manifest_name "manifest" \
--encodec_folder_name ${encodec_codes_folder_name} \
--dataset $dataset \
--exp_dir "${exp_root}/${dataset}/${exp_name}" \
--dataset_dir ${dataset_dir} \
--load_model_from ${load_model_path}