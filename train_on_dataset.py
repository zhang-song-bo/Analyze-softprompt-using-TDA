import logging
import json
import os
import torch
from torch import Tensor
from tqdm.auto import tqdm
from dataclasses import dataclass
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse

import utils

# 配置日志
logger = logging.getLogger("softprompts")
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s [%(filename)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


# --- Softprompts 库的配置---
@dataclass
class SoftPromptConfig:
    num_steps: int = 10
    num_epochs: int = 1
    batch_size: int = 5
    optim_str_init: str = "x x x x x x x x x x x x x x x x x x x x"
    lr: float = 1e-6
    seed: int | None = None
    verbose: bool = True


def run_inference_and_record(model, tokenizer, questions, correct_answers, softprompt_embeds, epoch, results_file_path):
    """对此脚本的推理过程进行封装"""
    inference_batch_size = 4
    predictions = utils.generate_with_softprompt(
        model, tokenizer, questions, softprompt_embeds, max_new_tokens=400, batch_size=inference_batch_size
    )
    with open(results_file_path, 'w', encoding='utf-8') as f:
        f.write(f"--- Epoch {epoch} ---\n\n")
        for i, (question, correct_answer, prediction) in enumerate(zip(questions, correct_answers, predictions)):
            f.write(f"--- Data Entry {i + 1} ---\n")
            f.write(f"Question: {question}\n")
            f.write(f"Correct Answer: {correct_answer}\n")
            f.write(f"Prediction: {prediction}\n\n")
    print(f"Epoch {epoch}: 推理结果已记录至 {results_file_path}")
    return predictions


# --- 用户的主训练脚本 ---
if __name__ == "__main__":
    # --- 1. NEW: 配置命令行参数 ---
    parser = argparse.ArgumentParser(description="Train a soft prompt on a full dataset.")
    parser.add_argument('--dataset_path', type=str, default="../data/dataset/BBH.jsonl",
                        help='Path to the full dataset file (.jsonl).')
    parser.add_argument('--model_save_dir', type=str, default="../models/",
                        help='Directory to save soft prompt checkpoints.')
    parser.add_argument('--results_save_dir', type=str, default="../results/",
                        help='Directory to save inference results and plots.')
    parser.add_argument('--model_name', type=str, default="google/gemma-2b-it",
                        help='Name of the pre-trained model.')
    parser.add_argument('--epochs', type=int, default=300,
                        help='Number of training epochs.')
    parser.add_argument('--save_every', type=int, default=2,
                        help='Save a checkpoint every N epochs.')
    parser.add_argument('--batch_size', type=int, default=5,
                        help='Physical batch size for training.')
    parser.add_argument('--accumulation_steps', type=int, default=2,
                        help='Number of gradient accumulation steps.')
    parser.add_argument('--lr', type=float, default=5e-6,
                        help='Learning rate.')
    parser.add_argument('--optim_str_init', type=str, default="x " * 40,
                        help='Initial string for the soft prompt.')

    args = parser.parse_args()

    # --- 2. 使用args中的参数进行初始化 ---
    os.makedirs(args.model_save_dir, exist_ok=True)
    os.makedirs(args.results_save_dir, exist_ok=True)

    print(f"加载模型与分词器: {args.model_name}")
    model, tokenizer = utils.get_model_and_tokenizer(args.model_name, device_map="cuda")
    print("模型与分词器加载完毕。")

    try:
        dataset = utils.load_data_from_jsonl(args.dataset_path)
    except (FileNotFoundError, ValueError) as e:
        print(f"加载数据时出错: {e}")
        exit()

    all_questions = [entry['question'] for entry in dataset]
    all_correct_answers = [entry['answer'] for entry in dataset]

    print(f"开始为 {len(dataset)} 条数据训练 Soft Prompt...")
    print(
        f"物理批次大小: {args.batch_size}, 梯度累积步数: {args.accumulation_steps}, "
        f"有效批次大小: {args.batch_size * args.accumulation_steps}")

    # --- 3. Soft Prompt 训练循环配置 ---
    config = SoftPromptConfig(
        num_epochs=1,  # 外部循环控制总epochs
        batch_size=args.batch_size,
        optim_str_init=args.optim_str_init,
        lr=args.lr
    )
    training_data = list(zip(all_questions, all_correct_answers))
    dataloader = torch.utils.data.DataLoader(
        training_data, batch_size=config.batch_size, shuffle=True
    )
    optim_ids, _ = utils.tokenize(
        tokenizer, config.optim_str_init, add_special_tokens=False
    )
    optim_ids = optim_ids.to(model.device)
    optim_embeds = (
        model.get_input_embeddings()(optim_ids)
        .detach()
        .clone()
        .requires_grad_()
    )
    optimizer_eps = (
        1e-6 if model.dtype in [torch.float16, torch.bfloat16] else 1e-8
    )
    optimizer = torch.optim.Adam(
        [optim_embeds], lr=config.lr, eps=optimizer_eps
    )
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)

    training_losses = []

    # --- 4. 训练循环 ---
    for epoch in range(1, args.epochs + 1):
        print(f"\n--- 训练 Epoch {epoch} ---")
        total_loss = 0
        optimizer.zero_grad()

        for batch_idx, (messages_batch, target_batch) in enumerate(tqdm(
                dataloader,
                desc=f"Epoch {epoch}/{args.epochs}",
                disable=not config.verbose,
        )):
            messages_batch = utils.add_optim_token_str_at_end(list(messages_batch))
            input_ids, attn_mask = utils.tokenize(
                tokenizer,
                messages_batch,
                add_chat_template=True,
                padding="longest",
                padding_side="left",
            )
            input_ids = input_ids.to(model.device)
            attn_mask = attn_mask.to(model.device)
            left_input_ids, right_input_ids, left_attn_mask, right_attn_mask = (
                utils.split_tokenized_messages_on_optim_str(
                    input_ids, attn_mask, tokenizer.optim_token_id
                )
            )
            target_ids, target_attn_mask = utils.tokenize(
                tokenizer,
                list(target_batch),
                padding="longest",
                padding_side="right",
                add_special_tokens=False,
            )
            target_ids = target_ids.to(model.device)
            target_attn_mask = target_attn_mask.to(model.device)
            before_embeds, after_embeds, target_embeds = [
                model.get_input_embeddings()(ids)
                for ids in (left_input_ids, right_input_ids, target_ids)
            ]
            batch_size_actual = input_ids.shape[0]

            batched_optim_embeds = optim_embeds.expand(batch_size_actual, -1, -1)
            batched_attn_mask = torch.ones(
                batch_size_actual, optim_embeds.shape[1], device=model.device
            )
            input_embeds = torch.cat(
                [
                    before_embeds.detach(),
                    batched_optim_embeds,
                    after_embeds.detach(),
                    target_embeds.detach(),
                ],
                dim=1,
            )
            input_attn_mask = torch.cat(
                [
                    left_attn_mask.detach(),
                    batched_attn_mask,
                    right_attn_mask.detach(),
                    target_attn_mask.detach(),
                ],
                dim=1,
            )
            logits = model(
                inputs_embeds=input_embeds,
                attention_mask=input_attn_mask,
            ).logits
            shift = input_embeds.shape[1] - target_ids.shape[1]
            shift_logits = logits[..., shift - 1: -1, :].contiguous()
            loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                target_ids.view(-1),
            )

            loss = loss / args.accumulation_steps
            loss.backward()
            torch.nn.utils.clip_grad_norm_(optim_embeds, max_norm=1.0)

            if (batch_idx + 1) % args.accumulation_steps == 0 or (batch_idx + 1) == len(dataloader):
                optimizer.step()
                optimizer.zero_grad()

            loss_float = loss.item() * args.accumulation_steps
            total_loss += loss_float

        avg_loss = total_loss / len(dataloader)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch} 平均损失: {avg_loss:.4f}, 当前学习率: {current_lr:.2e}")
        training_losses.append(avg_loss)
        scheduler.step(avg_loss)

        if epoch % args.save_every == 0:
            save_path = os.path.join(args.model_save_dir, f"soft_prompt_epoch_{epoch:03d}.pt")
            torch.save(optim_embeds.detach().cpu(), save_path)
            print(f"Epoch {epoch}: Soft Prompt 已保存至 {save_path}")

            with torch.no_grad():
                inference_output_filename = f"inference_results_epoch_{epoch:03d}.txt"
                inference_output_filepath = os.path.join(args.results_save_dir, inference_output_filename)
                run_inference_and_record(
                    model, tokenizer, all_questions, all_correct_answers,
                    optim_embeds, epoch, inference_output_filepath
                )

    print("Soft Prompt 在整个数据集上的训练与推理已完成。")

    # --- 5. 损失值绘制 ---
    print("\n正在绘制训练损失曲线...")
    plt.figure(figsize=(14, 8))
    plt.plot(range(1, len(training_losses) + 1), training_losses, marker='o', linestyle='-')
    plt.title('模型训练损失曲线')
    plt.xlabel('Epoch')
    plt.ylabel('平均损失')
    plt.grid(True)
    loss_curve_path = os.path.join(args.results_save_dir, "training_loss_curve.png")
    plt.savefig(loss_curve_path)
    plt.show()
    print(f"损失曲线图已保存至 {loss_curve_path}")