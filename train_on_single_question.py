import os
import torch
from tqdm.auto import tqdm
from transformers import set_seed
import matplotlib.pyplot as plt
import argparse

# 从utils模块导入所有共享函数
import utils

# --- 主训练脚本 ---
if __name__ == "__main__":
    # --- 1. NEW: 配置命令行参数 ---
    parser = argparse.ArgumentParser(description="Train a soft prompt on a single question.")
    parser.add_argument('--data_path', type=str, default="../data/single/single_HotpotQA_data2.jsonl", help='Path to the single question data file.')
    parser.add_argument('--model_save_dir', type=str, default="../models/", help='Directory to save soft prompt checkpoints.')
    parser.add_argument('--results_save_dir', type=str, default="../results/", help='Directory to save inference results and plots.')
    parser.add_argument('--model_name', type=str, default="google/gemma-2b-it", help='Name of the pre-trained model.')
    parser.add_argument('--epochs', type=int, default=300, help='Number of training epochs.')
    parser.add_argument('--save_every', type=int, default=2, help='Save a checkpoint every N epochs.')
    parser.add_argument('--lr', type=float, default=5e-6, help='Learning rate.')
    parser.add_argument('--seed', type=int, default=101, help='Random seed for reproducibility.')
    parser.add_argument('--optim_str_init', type=str, default="x " * 40, help='Initial string for the soft prompt.')

    args = parser.parse_args()

    # --- 2. 初始化 ---
    set_seed(args.seed)
    os.makedirs(args.model_save_dir, exist_ok=True)
    os.makedirs(args.results_save_dir, exist_ok=True)

    print(f"Loading model and tokenizer: {args.model_name}")
    model, tokenizer = utils.get_model_and_tokenizer(args.model_name, device_map="cuda") # <<< MODIFIED: 使用utils函数
    print("Model and tokenizer loaded.")

    try:
        single_data_entry = utils.load_single_question_data(args.data_path) # <<< MODIFIED: 使用utils函数
    except (ValueError, FileNotFoundError) as e:
        print(f"Error loading data: {e}")
        exit()

    question_text = single_data_entry['question']
    correct_answer = single_data_entry['answer']

    print(f"Starting Soft Prompt training for the single question...")

    # --- 3. 准备 Soft Prompt 和优化器 ---
    print(f"Initializing soft prompt with: '{args.optim_str_init}'")
    optim_ids, _ = utils.tokenize(tokenizer, args.optim_str_init) # <<< MODIFIED
    num_prompt_tokens = optim_ids.shape[1]
    print(f"Number of prompt tokens: {num_prompt_tokens}")

    optim_ids = optim_ids.to(model.device)
    optim_embeds = model.get_input_embeddings()(optim_ids).detach().clone().requires_grad_()

    optimizer_eps = 1e-6 if model.dtype in [torch.float16, torch.bfloat16] else 1e-8
    optimizer = torch.optim.Adam([optim_embeds], lr=args.lr, eps=optimizer_eps)

    embedding_dim = model.config.hidden_size
    positional_encoding = utils.get_sinusoidal_positional_encoding( # <<< MODIFIED
        num_prompt_tokens, embedding_dim, dtype=model.dtype
    ).to(model.device)

    message_dataloader = torch.utils.data.DataLoader([question_text], batch_size=1, shuffle=False)
    target_dataloader = torch.utils.data.DataLoader([correct_answer], batch_size=1, shuffle=False)

    training_losses = []

    # --- 4. 训练循环 ---
    for epoch in range(1, args.epochs + 1):
        epoch_total_loss = 0.0

        for messages_batch, target_batch in zip(message_dataloader, target_dataloader):
            optimizer.zero_grad()

            messages_with_optim = utils.add_optim_token_str_at_end(messages_batch) # <<< MODIFIED
            input_ids, attn_mask = utils.tokenize(tokenizer, messages_with_optim, add_chat_template=True, padding="longest", padding_side="left") # <<< MODIFIED
            input_ids, attn_mask = input_ids.to(model.device), attn_mask.to(model.device)
            left_input_ids, right_input_ids, left_attn_mask, right_attn_mask = utils.split_tokenized_messages_on_optim_str( # <<< MODIFIED
                input_ids, attn_mask, tokenizer.optim_token_id)

            target_ids, target_attn_mask = utils.tokenize(tokenizer, target_batch, padding="longest", add_special_tokens=False) # <<< MODIFIED
            target_ids, target_attn_mask = target_ids.to(model.device), target_attn_mask.to(model.device)

            before_embeds = model.get_input_embeddings()(left_input_ids).to(model.dtype)
            after_embeds = model.get_input_embeddings()(right_input_ids).to(model.dtype)
            target_embeds = model.get_input_embeddings()(target_ids).to(model.dtype)

            optim_embeds_with_pe = optim_embeds + positional_encoding
            batched_attn_mask = torch.ones(1, num_prompt_tokens, device=model.device)

            input_embeds = torch.cat([before_embeds, optim_embeds_with_pe, after_embeds, target_embeds], dim=1)
            input_attn_mask = torch.cat([left_attn_mask, batched_attn_mask, right_attn_mask, target_attn_mask], dim=1)

            logits = model(inputs_embeds=input_embeds, attention_mask=input_attn_mask).logits
            shift = input_embeds.shape[1] - target_ids.shape[1]
            shift_logits = logits[..., shift - 1: -1, :].contiguous()
            loss = torch.nn.functional.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), target_ids.view(-1))

            loss.backward()
            optimizer.step()
            epoch_total_loss += loss.item()

        print(f"Epoch {epoch}/{args.epochs}: Average Loss = {epoch_total_loss:.6f}")
        training_losses.append(epoch_total_loss)

        if epoch % args.save_every == 0 or epoch == args.epochs:
            save_path = os.path.join(args.model_save_dir, f"soft_prompt_epoch_{epoch:03d}.pt")
            torch.save(optim_embeds.detach().cpu(), save_path)
            print(f"Epoch {epoch}: Soft Prompt state saved to {save_path}")

            inference_output_filename = f"inference_results_epoch_{epoch:03d}.txt"
            inference_output_filepath = os.path.join(args.results_save_dir, inference_output_filename)
            
            # 封装推理逻辑为一个函数，使其更清晰
            def run_inference_and_record(epoch_num, output_path):
                generation = utils.generate_with_softprompt(
                    model, tokenizer, question_text, optim_embeds.detach(),
                    max_new_tokens=400,
                    positional_encoding=positional_encoding.detach().unsqueeze(0)
                )
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(f"--- Epoch {epoch_num} ---\n")
                    f.write(f"Question: {question_text}\n\n")
                    f.write(f"Correct Answer:\n{correct_answer}\n\n")
                    f.write(f"Prediction:\n{generation[0]}\n")
                print(f"Epoch {epoch_num}: Inference results recorded to {output_path}")

            run_inference_and_record(epoch, inference_output_filepath)

    print("\nSoft Prompt training completed.")

    # --- 5. 绘制损失曲线 ---
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, args.epochs + 1), training_losses, marker='.', linestyle='-')
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    loss_curve_path = os.path.join(args.results_save_dir, "training_loss_curve.png")
    plt.savefig(loss_curve_path)
    plt.show()
    print(f"Loss curve saved to {loss_curve_path}")