# @ hwang258@jh.edu

import os

from edit_utils_en import parse_edit_en
from edit_utils_zh import parse_edit_zh
from inference_v2 import WhisperxAlignModel, WhisperxModel, align, traditional_to_simplified, transcribe
os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ["USER"] = "root" # TODO change this to your username

import shutil
import torch
import torchaudio
import numpy as np
import random
from argparse import Namespace
from data.tokenizer import (
    AudioTokenizer,
    TextTokenizer,
)
import torchaudio
import torchaudio.transforms as transforms
from inference_scale import inference_one_sample
import time
from tqdm import tqdm
import argparse
from models import ssr
import re
import uuid
import opencc

def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"using {device}")

def get_mask_interval(transcribe_state, word_span):
    seg_num = len(transcribe_state['segments'])
    data = []
    for i in range(seg_num):
        words = transcribe_state['segments'][i]['words']
        for item in words:
          data.append([item['start'], item['end'], item['word']])

    s, e = word_span[0], word_span[1]
    assert s <= e, f"s:{s}, e:{e}"
    assert s >= 0, f"s:{s}"
    assert e <= len(data), f"e:{e}"
    if e == 0: # start
        start = 0.
        end = float(data[0][0])
    elif s == len(data): # end
        start = float(data[-1][1])
        end = float(data[-1][1]) # don't know the end yet
    elif s == e: # insert
        start = float(data[s-1][1])
        end = float(data[s][0])
    else:
        start = float(data[s-1][1]) if s > 0 else float(data[s][0])
        end = float(data[e][0]) if e < len(data) else float(data[-1][1])

    return (start, end)

def parse_manifest_word_spans(word_span_str):
    """解析manifest中的word span字符串，转换为适合新推理函数的格式"""
    spans = word_span_str.split(",")
    if len(spans) == 2:
        return [int(spans[0]), int(spans[1])]
    else:
        # 如果是单个数字，假设是插入操作
        span_idx = int(spans[0])
        return [span_idx, span_idx]

def parse_args():
    parser = argparse.ArgumentParser(description="inference speech editing")
    # manifest相关参数
    parser.add_argument("--manifest_fn", type=str, required=True, help="path to manifest file")
    parser.add_argument("--audio_root", type=str, required=True, help="root path to audio files")
    parser.add_argument("--left_margin", type=float, default=0.08, help="extra space on the left to the word boundary")
    parser.add_argument("--right_margin", type=float, default=0.08, help="extra space on the right to the word boundary")
    
    # 原有参数
    parser.add_argument("--sub_amount", type=float, default=0.12, help="if the performance is not good, try modify this span, not used for tts")
    parser.add_argument('--codec_audio_sr', type=int, default=16000)
    parser.add_argument('--codec_sr', type=int, default=50)
    parser.add_argument('--top_k', type=int, default=0)
    parser.add_argument('--top_p', type=float, default=0.8)
    parser.add_argument('--temperature', type=int, default=1)
    parser.add_argument('--kvcache', type=int, default=1)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--stop_repetition', type=int, default=2)
    parser.add_argument('--sample_batch_size', type=int, default=1)
    parser.add_argument('--cfg_coef', type=float, default=1.5)
    parser.add_argument('--cfg_stride', type=int, default=1)
    parser.add_argument('--aug_text', action='store_true')
    parser.add_argument('--aug_context', action='store_true')
    parser.add_argument('--use_watermark', action='store_true')
    parser.add_argument('--tts', action='store_true')
    parser.add_argument('--prompt_length', type=int, default=3)
    parser.add_argument('--language', type=str, choices=["en", "zh"], help="choose from en or zh")
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--codec_path', type=str, default=None)
    parser.add_argument('--temp_folder', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--whisper_model_name', type=str, choices=["base.en", "base"], default="base.en")
    return parser.parse_args()


def main(args):
    seed_everything(args.seed)
    if args.language != 'en' and args.language != 'zh':
        raise RuntimeError("We only support English or Mandarin now!")
        
    # 初始化模型
    filepath = os.path.join(args.model_path)
    ckpt = torch.load(filepath, map_location="cpu", weights_only=False)
    model = ssr.SSR_Speech(ckpt["config"])
    model.load_state_dict(ckpt["model"])
    config = vars(model.args)
    phn2num = ckpt["phn2num"]
    model.to(device)
    model.eval()
    
    # 初始化tokenizer
    audio_tokenizer = AudioTokenizer(signature=args.codec_path)
    text_tokenizer = TextTokenizer(backend="espeak") if args.language == 'en' else TextTokenizer(backend="espeak", language='cmn')
    
    align_model = WhisperxAlignModel(args.language)
    transcribe_model = WhisperxModel(args.whisper_model_name, align_model, args.language)
    
    # 创建必要的目录
    os.makedirs(args.temp_folder, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 读取manifest文件，与原脚本相同的方式
    with open(args.manifest_fn, "r") as rf:
        manifest = [l.strip().split("\t") for l in rf.readlines()]
    manifest = manifest[1:]  # 跳过标题行
    
    audio_fns = []
    target_texts = []
    orig_texts = []
    mask_intervals = []
    edit_types = []
    new_spans = []
    orig_spans = []
    
    start_time = time.time()
    
    # 解析manifest，与原脚本相同的逻辑
    for item in manifest:
        audio_fn = os.path.join(args.audio_root, item[0])
        temp = torchaudio.info(audio_fn)
        audio_dur = temp.num_frames/temp.sample_rate
        audio_fns.append(audio_fn)
        target_text = item[2].split("|")[-1]
        orig_text = item[1].split("|")[0]
        edit_types.append(item[5].split("|"))
        new_spans.append(item[4].split("|"))
        orig_spans.append(item[3].split("|"))
        target_texts.append(target_text) # the last transcript is the target
        orig_texts.append(orig_text)
        
        audio_name_wo_snr = item[0].split('_snr')
        if len(audio_name_wo_snr) > 1:
            audio_name_wo_snr = audio_name_wo_snr[0] + ".wav"
        # alignment_fn = os.path.join(args.audio_root, "aligned", audio_name_wo_snr.replace(".wav", ".csv"))
        # if not os.path.isfile(alignment_fn):
        #     alignment_fn = alignment_fn.replace("/aligned/", "/aligned_csv/")
        #     assert os.path.isfile(alignment_fn), alignment_fn
        
        

    # 批量处理每个样本
    for i, (audio_fn, target_text) in enumerate(tqdm(zip(audio_fns, target_texts))):
        # 复制音频到临时文件夹
        filename = os.path.splitext(os.path.basename(audio_fn))[0]
        temp_audio_fn = f"{args.temp_folder}/{filename}.wav"
        
        # 重采样到16kHz
        import librosa
        import soundfile as sf
        audio, _ = librosa.load(audio_fn, sr=16000)
        sf.write(temp_audio_fn, audio, 16000)
        
        # 处理原始文本和目标文本
        
        orig_transcript, segments = transcribe(audio_fn, transcribe_model)
        # 处理语言特定的文本
        if args.language == 'zh':
            converter = opencc.OpenCC('t2s')
            orig_transcript = converter.convert(orig_transcript)
            transcribe_state = align(args, traditional_to_simplified(segments), audio_fn, align_model)
            transcribe_state['segments'] = traditional_to_simplified(transcribe_state['segments'])
        elif args.language == 'en':
            orig_transcript = orig_transcript.lower()
            target_transcript = target_text.lower()
            transcribe_state = align(args, segments, audio_fn, align_model)
        
        # run the script to turn user input to the format that the model can take
        if not args.tts:
            operations, orig_spans = parse_edit_en(orig_transcript, target_transcript) if args.language == 'en' else parse_edit_zh(orig_transcript,             operations, orig_spans = parse_edit_en(orig_transcript, target_transcript) if args.language == 'en' else parse_edit_zh(orig_transcript, target_transcript)
)
            print(operations)
            print("orig_spans: ", orig_spans)
            
            if len(orig_spans) > 3:
                raise RuntimeError("Current model only supports maximum 3 editings")
                
            starting_intervals = []
            ending_intervals = []
            for orig_span in orig_spans:
                start, end = get_mask_interval(transcribe_state, orig_span)
                starting_intervals.append(start)
                ending_intervals.append(end)
        
            print("intervals: ", starting_intervals, ending_intervals)
        
            info = torchaudio.info(audio_fn)
            audio_dur = info.num_frames / info.sample_rate
            
            def combine_spans(spans, threshold=0.2):
                spans.sort(key=lambda x: x[0])
                combined_spans = []
                current_span = spans[0]
        
                for i in range(1, len(spans)):
                    next_span = spans[i]
                    if current_span[1] >= next_span[0] - threshold:
                        current_span[1] = max(current_span[1], next_span[1])
                    else:
                        combined_spans.append(current_span)
                        current_span = next_span
                combined_spans.append(current_span)
                return combined_spans
            
            morphed_span = [[max(start - args.sub_amount, 0), min(end + args.sub_amount, audio_dur)]
                            for start, end in zip(starting_intervals, ending_intervals)] # in seconds
            morphed_span = combine_spans(morphed_span, threshold=0.2)
            print("morphed_spans: ", morphed_span)
            # save_morphed_span = f"{args.output_dir}/{audio_fn}_mask.pt"
            # torch.save(morphed_span, save_morphed_span)
            mask_interval = [[round(span[0]*args.codec_sr), round(span[1]*args.codec_sr)] for span in morphed_span]
            mask_interval = torch.LongTensor(mask_interval) # [M,2], M==1 for now
        else:
            info = torchaudio.info(audio_fn)
            audio_dur = info.num_frames / info.sample_rate
            
            morphed_span = [(audio_dur, audio_dur)] # in seconds
            mask_interval = [[round(span[0]*args.codec_sr), round(span[1]*args.codec_sr)] for span in morphed_span]
            mask_interval = torch.LongTensor(mask_interval) # [M,2], M==1 for now
            print("mask_interval: ", mask_interval)
        
        print(f"Processing sample {i+1}/{len(audio_fns)}: {os.path.basename(audio_fn)}")
        print(f"Target text: {target_transcript}")
        print(f"Prompt text: {orig_text}")
        print(f"Mask intervals: {mask_interval}")
        
        decode_config = {'top_k': args.top_k, 'top_p': args.top_p, 'temperature': args.temperature, 
                         'stop_repetition': args.stop_repetition, 'kvcache': args.kvcache, 
                         "codec_audio_sr": args.codec_audio_sr, "codec_sr": args.codec_sr}
        
        # 对每个样本进行多次推理
        for num in range(args.sample_batch_size):
            seed_everything(args.seed + num)
            
            # 使用新的inference函数
            new_audio = inference_one_sample(
                model, Namespace(**config), phn2num, text_tokenizer, audio_tokenizer, 
                temp_audio_fn, orig_text, target_transcript, mask_interval, 
                args.cfg_coef, args.cfg_stride, args.aug_text, args.aug_context, 
                args.use_watermark, args.tts, device, decode_config
            )
            
            # 保存生成的音频
            new_audio = new_audio[0].cpu()
            save_fn_new = f"{args.output_dir}/{filename}_new_seed{args.seed+num}.wav"
            torchaudio.save(save_fn_new, new_audio, args.codec_audio_sr)
            
            print(f"Generated: {save_fn_new}")
        
        # 保存原始音频
        save_fn_orig = f"{args.output_dir}/{filename}_orig.wav"
        if not os.path.isfile(save_fn_orig):
            orig_audio, orig_sr = torchaudio.load(audio_fn)
            if orig_sr != args.codec_audio_sr:
                orig_audio = torchaudio.transforms.Resample(orig_sr, args.codec_audio_sr)(orig_audio)
            torchaudio.save(save_fn_orig, orig_audio, args.codec_audio_sr)
        
        # 清理临时文件
        os.remove(temp_audio_fn)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\nTotal running time: {elapsed_time:.4f} s")


if __name__ == "__main__":
    args = parse_args()
    main(args)