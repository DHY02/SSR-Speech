#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
词汇表合并脚本
合并新数据集的vocab.txt和预训练模型的vocab_en.txt
"""

import os
import argparse
import logging
from collections import OrderedDict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="合并新旧词汇表")
    parser.add_argument('--new_vocab_path', type=str, required=True,
                       help="新数据集的vocab.txt路径")
    parser.add_argument('--old_vocab_path', type=str, required=True,
                       help="预训练模型的vocab_en.txt路径")
    parser.add_argument('--output_path', type=str, required=True,
                       help="输出合并后词汇表的路径")
    parser.add_argument('--target_vocab_size', type=int, default=100,
                       help="目标词汇表大小，建议英文100，中文200")
    return parser.parse_args()

def load_vocab_file(file_path):
    """加载词汇表文件，返回 {phoneme: index} 字典"""
    vocab = {}
    if not os.path.exists(file_path):
        logger.error(f"文件不存在: {file_path}")
        return vocab
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f.readlines()):
                line = line.strip()
                if not line:
                    continue
                parts = line.split(' ', 1)
                if len(parts) == 2:
                    idx, phoneme = parts
                    try:
                        vocab[phoneme] = int(idx)
                    except ValueError:
                        logger.warning(f"第{line_num+1}行格式错误: {line}")
                else:
                    logger.warning(f"第{line_num+1}行格式错误: {line}")
        logger.info(f"从 {file_path} 加载了 {len(vocab)} 个音素")
    except Exception as e:
        logger.error(f"读取文件失败 {file_path}: {e}")
    
    return vocab

def merge_vocabs(old_vocab, new_vocab, target_size):
    """
    合并词汇表的策略：
    1. 保留旧词汇表中的所有音素和索引（保持兼容性）
    2. 为新音素分配新的索引
    3. 确保不超过目标大小
    """
    merged_vocab = OrderedDict()
    
    # 首先添加旧词汇表中的所有音素
    for phoneme, idx in sorted(old_vocab.items(), key=lambda x: x[1]):
        merged_vocab[phoneme] = idx
    
    # 找出新音素（在新词汇表中但不在旧词汇表中的）
    old_phonemes = set(old_vocab.keys())
    new_phonemes = set(new_vocab.keys())
    
    common_phonemes = old_phonemes & new_phonemes
    truly_new_phonemes = new_phonemes - old_phonemes
    missing_phonemes = old_phonemes - new_phonemes
    
    logger.info(f"旧词汇表: {len(old_phonemes)} 个音素")
    logger.info(f"新词汇表: {len(new_phonemes)} 个音素")
    logger.info(f"共同音素: {len(common_phonemes)} 个")
    logger.info(f"新增音素: {len(truly_new_phonemes)} 个")
    logger.info(f"缺失音素: {len(missing_phonemes)} 个")
    
    if truly_new_phonemes:
        logger.info(f"新增音素列表: {sorted(list(truly_new_phonemes))}")
    
    if missing_phonemes:
        logger.warning(f"旧词汇表中有但新数据中没有的音素: {sorted(list(missing_phonemes))}")
        logger.warning("这些音素将保留在最终词汇表中以保持模型兼容性")
    
    # 为新音素分配索引
    if truly_new_phonemes:
        # 找到当前最大索引
        max_idx = max(old_vocab.values()) if old_vocab else -1
        next_idx = max_idx + 1
        
        for phoneme in sorted(truly_new_phonemes):
            if len(merged_vocab) < target_size:
                merged_vocab[phoneme] = next_idx
                next_idx += 1
                logger.info(f"为新音素 '{phoneme}' 分配索引 {next_idx-1}")
            else:
                logger.warning(f"词汇表已达到目标大小 {target_size}，无法添加音素: {phoneme}")
                break
    
    return merged_vocab

def save_vocab(vocab, output_path):
    """保存词汇表到文件"""
    # 按索引排序
    sorted_items = sorted(vocab.items(), key=lambda x: x[1])
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, (phoneme, idx) in enumerate(sorted_items):
                if i < len(sorted_items) - 1:
                    f.write(f"{idx} {phoneme}\n")
                else:
                    f.write(f"{idx} {phoneme}")
        logger.info(f"合并后的词汇表已保存到: {output_path}")
    except Exception as e:
        logger.error(f"保存词汇表失败: {e}")

def main():
    args = parse_args()
    
    logger.info("开始合并词汇表...")
    logger.info(f"旧词汇表: {args.old_vocab_path}")
    logger.info(f"新词汇表: {args.new_vocab_path}")
    logger.info(f"输出路径: {args.output_path}")
    logger.info(f"目标大小: {args.target_vocab_size}")
    
    # 加载词汇表
    old_vocab = load_vocab_file(args.old_vocab_path)
    new_vocab = load_vocab_file(args.new_vocab_path)
    
    if not old_vocab and not new_vocab:
        logger.error("无法加载任何词汇表")
        return
    
    # 合并词汇表
    merged_vocab = merge_vocabs(old_vocab, new_vocab, args.target_vocab_size)
    
    # 检查最终大小
    final_size = len(merged_vocab)
    if final_size > args.target_vocab_size:
        logger.error(f"合并后词汇表大小 {final_size} 超过目标大小 {args.target_vocab_size}")
        logger.error("建议增加 target_vocab_size 参数")
        return
    
    # 保存合并后的词汇表
    save_vocab(merged_vocab, args.output_path)
    
    # 生成配置建议
    recommended_vocab_size = max(args.target_vocab_size, final_size)
    
    logger.info("="*60)
    logger.info("合并完成！统计信息：")
    logger.info(f"最终词汇表大小: {final_size}")
    logger.info(f"建议的训练配置:")
    logger.info(f"  --text_vocab_size {recommended_vocab_size}")
    logger.info(f"  --text_pad_token {recommended_vocab_size}")
    logger.info("="*60)
    
    # 验证词汇表
    logger.info("验证词汇表...")
    test_vocab = load_vocab_file(args.output_path)
    if len(test_vocab) == final_size:
        logger.info("✓ 词汇表验证通过")
    else:
        logger.error("✗ 词汇表验证失败")
    
    # 显示前几个和后几个条目作为示例
    sorted_items = sorted(merged_vocab.items(), key=lambda x: x[1])
    logger.info("词汇表示例（前5个）:")
    for phoneme, idx in sorted_items[:5]:
        logger.info(f"  {idx}: {phoneme}")
    
    if len(sorted_items) > 10:
        logger.info("词汇表示例（后5个）:")
        for phoneme, idx in sorted_items[-5:]:
            logger.info(f"  {idx}: {phoneme}")

if __name__ == "__main__":
    main()