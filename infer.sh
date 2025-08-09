export HF_ENDPOINT=https://hf-mirror.com
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 
python inference_editing_scale.py \
    --manifest_fn "./mix_expanded.txt" \
    --audio_root "/tsdata2/dhy/data/realedit_music_mixed" \
    --seed 2024 \
    --aug_text \
    --use_watermark \
    --language 'en' \
    --model_path "../../models/SSR-Speech-English/English.pth" \
    --codec_path "../../models/SSR-Speech-English/wmencodec.th" \
    --left_margin 0.08 \
    --right_margin 0.08 \
    --cfg_coef 1.5 \
    --cfg_stride 5 \
    --sub_amount 0.12 \
    --temp_folder "./demo/temp" \
    --output_dir "./demo/generated_se"