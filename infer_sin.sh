export HF_ENDPOINT=https://hf-mirror.com
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 
python inference_v2.py  \
    --seed 2024 \
    --cfg_stride 5 \
    --sub_amount 0.12 \
    --aug_text \
    --use_watermark \
    --language 'en' \
    --model_path "../../models/SSR-Speech-English/English.pth" \
    --codec_path "../../models/SSR-Speech-English/wmencodec.th" \
    --orig_audio "./demo/84_121550_000074_000000.wav" \
    --target_transcript "But when I saw the mirage of the lake in the distance, which the sense deceives, Lost not by distance any marks," \
    --temp_folder "./demo/temp" \
    --output_dir "./demo/generated_se" \
    --savename "84_121550_000074_00000" \
    --whisper_model_name "base.en"