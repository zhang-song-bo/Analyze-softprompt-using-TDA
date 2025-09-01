# track_accuracy_over_epochs.py

import torch
import os
import re
from tqdm.auto import tqdm
from datetime import datetime
import argparse  # <<< NEW: 引入argparse

# <<< NEW: 从utils模块导入所有共享函数
import utils

# =============================== 主程序 ===============================
if __name__ == "__main__":
    # --- 1. NEW: 配置命令行参数 ---
    parser = argparse.ArgumentParser(description="Evaluate all soft prompt checkpoints on a single question.")
    parser.add_argument('--prompt_dir', type=str, default="../models_pe_prox/",
                        help='Directory containing all soft prompt .pt files.')
    parser.add_argument('--data_path', type=str, default="../data/single/single_gsm8k_data.jsonl",
                        help='Path to the single question data file for evaluation.')
    parser.add_argument('--output_log_path', type=str, default="../results_pe_prox/full_evaluation_log.txt",
                        help='Path to save the full evaluation log file.')
    parser.add_argument('--model_name', type=str, default="google/gemma-2b-it",
                        help='Name of the base model for evaluation.')

    args = parser.parse_args()

    print(f"--- 开始对所有模型进行评估并记录输出 ---")

    # --- 步骤 1: 检查输入路径是否存在 ---
    if not os.path.isdir(args.prompt_dir):
        print(f"错误: 文件夹未找到 '{args.prompt_dir}'")
        exit()
    if not os.path.exists(args.data_path):
        print(f"错误: 数据文件未找到 '{args.data_path}'")
        exit()

    # --- 步骤 2: 加载模型、分词器和数据 ---
    print(f"加载基础模型和分词器: {args.model_name}...")
    model, tokenizer = utils.get_model_and_tokenizer(args.model_name, device_map="cuda")  # <<< MODIFIED
    print("加载问题数据...")
    single_data = utils.load_single_question_data(args.data_path)  # <<< MODIFIED
    question_text = single_data['question']

    # --- 步骤 3: 获取并排序所有模型文件 ---
    print(f"正在从 '{args.prompt_dir}' 读取所有模型文件...")
    all_files = sorted(
        [f for f in os.listdir(args.prompt_dir) if f.endswith(".pt")],
        key=lambda f: int(re.search(r'(\d+)', f).group(1))
    )

    if not all_files:
        print(f"错误: 在 '{args.prompt_dir}' 目录下没有找到 .pt 文件。")
        exit()

    print(f"找到 {len(all_files)} 个模型文件，将进行测试并记录结果...")

    # --- 步骤 4: 循环测试并记录日志 ---
    os.makedirs(os.path.dirname(args.output_log_path), exist_ok=True)
    with open(args.output_log_path, 'w', encoding='utf-8') as log_file:
        log_file.write(f"评估开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write(f"评估模型文件夹: {args.prompt_dir}\n")
        log_file.write(f"评估问题文件: {args.data_path}\n")
        log_file.write(f"总计 {len(all_files)} 个模型文件待评估。\n{'=' * 80}\n\n")

        correct_count = 0
        with torch.no_grad():
            for filename in tqdm(all_files, desc="批量评估中"):
                prompt_path = os.path.join(args.prompt_dir, filename)
                soft_prompt_embeds = torch.load(prompt_path, map_location=model.device)

                num_prompt_tokens = soft_prompt_embeds.shape[1]
                embedding_dim = model.config.hidden_size
                positional_encoding = utils.get_sinusoidal_positional_encoding(  # <<< MODIFIED
                    num_prompt_tokens, embedding_dim, dtype=model.dtype
                ).to(model.device).unsqueeze(0)

                prediction_list = utils.generate_with_softprompt(  # <<< MODIFIED
                    model=model,
                    tokenizer=tokenizer,
                    messages=[question_text],
                    optim_embeds=soft_prompt_embeds,
                    positional_encoding=positional_encoding,
                    max_new_tokens=300,
                )
                prediction = prediction_list[0]

                correct = utils.is_correct(prediction, single_data)  # <<< MODIFIED
                status = "[正确]" if correct else "[错误]"
                if correct:
                    correct_count += 1

                print(f"  -> {filename}: {status}")

                log_entry = f"--- 模型文件: {filename} ---\n状态: {status}\n\n模型输出:\n{prediction}\n{'-' * 80}\n\n"
                log_file.write(log_entry)

        accuracy = (correct_count / len(all_files)) * 100 if all_files else 0
        summary = f"\n{'=' * 80}\n评估完成总结:\n总共测试模型数: {len(all_files)}\n正确回答数: {correct_count}\n错误回答数: {len(all_files) - correct_count}\n正确率: {accuracy:.2f}%\n{'=' * 80}\n"
        log_file.write(summary)

    print("\n\n--- 所有评估已完成 ---")
    print(f"详细的评估日志已保存至: {args.output_log_path}")