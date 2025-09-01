import torch
import transformers
import json
import re
import numpy as np
from jaxtyping import Int
from torch import Tensor
import logging
from tqdm.auto import tqdm

# --- 配置日志 ---
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


def set_tokenizer_optim_token(
        tokenizer: transformers.PreTrainedTokenizer,
        optim_token_str: str = "<|optim_str|>",
) -> None:
    """为分词器添加并设置用于优化的特殊token。"""
    tokenizer.add_special_tokens({"additional_special_tokens": [optim_token_str]})
    tokenizer.optim_token_id = tokenizer.convert_tokens_to_ids(optim_token_str)


def get_model_and_tokenizer(
        model_name_or_path: str,
        optim_token_str: str = "<|optim_str|>",
        device_map: str = "auto",
        **model_kwargs,
) -> tuple[transformers.PreTrainedModel, transformers.PreTrainedTokenizer]:
    """加载预训练模型和分词器，并进行必要的配置。"""
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.float16,
        device_map=device_map,
        **model_kwargs
    )
    for param in model.parameters():
        param.requires_grad = False

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path)
    set_tokenizer_optim_token(tokenizer, optim_token_str)

    tokenizer.padding_side = "left"
    if tokenizer.pad_token:
        pass
    elif tokenizer.unk_token:
        tokenizer.pad_token_id = tokenizer.unk_token_id
    elif tokenizer.eos_token:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    else:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer


def add_optim_token_str_at_end(
        messages: str | list[str], optim_token_str: str = "<|optim_str|>"
) -> list[str]:
    """在消息末尾添加优化token字符串。"""
    if isinstance(messages, str):
        messages = [messages]
    return [x + optim_token_str for x in messages]


def tokenize(
        tokenizer: transformers.PreTrainedTokenizer,
        text: str | list[str],
        add_chat_template: bool = False,
        **tokenizer_kwargs,
) -> tuple[Int[Tensor, "batch seq_len"], Int[Tensor, "batch seq_len"]]:
    """对文本进行分词。"""
    if isinstance(text, str):
        text = [text]

    if add_chat_template:
        batched_messages = [[{"role": "user", "content": msg}] for msg in text]
        text = tokenizer.apply_chat_template(
            batched_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        if tokenizer.bos_token:
            text = [
                t.replace(tokenizer.bos_token, "")
                for t in text
                if t.startswith(tokenizer.bos_token)
            ]

    inputs = tokenizer(
        text=text,
        return_tensors="pt",
        **tokenizer_kwargs,
    )
    return inputs["input_ids"], inputs["attention_mask"]


def split_tokenized_messages_on_optim_str(
        input_ids: Int[Tensor, "batch seq_len"],
        attn_mask: Int[Tensor, "batch seq_len"],
        optim_token_id: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """根据优化token的位置分割tokenized id和attention mask。"""
    optim_str_positions = (input_ids == optim_token_id).nonzero(as_tuple=False)
    unique_columns = torch.unique(optim_str_positions[:, 1])
    counts = torch.sum(input_ids == optim_token_id, dim=1)
    assert torch.all(counts == 1), "Not exactly one instance of optim_token_id in each row."
    assert len(unique_columns) == 1, "optim_token_id is not in the same column in all rows."

    insertion_column = unique_columns.item()
    left_input_ids = input_ids[:, :insertion_column]
    right_input_ids = input_ids[:, insertion_column + 1:]
    left_attn_mask = attn_mask[:, :insertion_column]
    right_attn_mask = attn_mask[:, insertion_column + 1:]
    return left_input_ids, right_input_ids, left_attn_mask, right_attn_mask


def generate_with_softprompt(
        model: transformers.PreTrainedModel,
        tokenizer: transformers.PreTrainedTokenizer,
        messages: str | list[str],
        optim_embeds: Tensor,
        optim_token_str: str = "<|optim_str|>",
        batch_size: int = 8,
        positional_encoding: Tensor | None = None,
        **generation_kwargs,
) -> str | list[str]:
    """使用soft prompt进行文本生成。"""
    if isinstance(messages, str):
        messages = [messages]

    if tokenizer.optim_token_id is None:
        logger.warning("Tokenizer does not have an optimization token id. Adding it manually.")
        set_tokenizer_optim_token(tokenizer, optim_token_str)

    if model.device == torch.device("cpu"):
        logger.warning("Running on CPU -- this will be slow!")

    dataloader = torch.utils.data.DataLoader(messages, batch_size=batch_size, shuffle=False)

    generations_list = []
    for batched_messages in tqdm(dataloader, desc="Generating", total=len(dataloader)):
        batched_messages = add_optim_token_str_at_end(batched_messages, optim_token_str)
        input_ids, attn_mask = tokenize(
            tokenizer, batched_messages, add_chat_template=True, padding="longest", padding_side="left"
        )
        input_ids = input_ids.to(model.device)
        attn_mask = attn_mask.to(model.device)
        left_input_ids, right_input_ids, left_attn_mask, right_attn_mask = split_tokenized_messages_on_optim_str(
            input_ids, attn_mask, tokenizer.optim_token_id
        )

        before_embeds, after_embeds = [model.get_input_embeddings()(ids) for ids in (left_input_ids, right_input_ids)]

        optim_embeds_with_pe = optim_embeds.to(model.device)
        if positional_encoding is not None:
            optim_embeds_with_pe = optim_embeds_with_pe + positional_encoding.to(model.device)

        batched_optim_embeds = optim_embeds_with_pe.expand(len(batched_messages), -1, -1)
        batched_attn_mask = torch.ones(len(batched_messages), optim_embeds.shape[1], device=model.device)

        input_embeds = torch.cat([before_embeds.detach(), batched_optim_embeds, after_embeds.detach()], dim=1)
        input_attn_mask = torch.cat([left_attn_mask.detach(), batched_attn_mask, right_attn_mask.detach()], dim=1)

        generation = model.generate(inputs_embeds=input_embeds, attention_mask=input_attn_mask, **generation_kwargs)
        generations_list.extend(generation)

    return tokenizer.batch_decode(generations_list, skip_special_tokens=True)


def get_sinusoidal_positional_encoding(num_tokens: int, embedding_dim: int, dtype=torch.float16) -> torch.Tensor:
    """生成正弦位置编码。"""
    positions = torch.arange(num_tokens, dtype=dtype).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, embedding_dim, 2, dtype=dtype) * -(np.log(10000.0) / embedding_dim))
    pe = torch.zeros(num_tokens, embedding_dim, dtype=dtype)
    pe[:, 0::2] = torch.sin(positions * div_term)
    pe[:, 1::2] = torch.cos(positions * div_term)
    return pe

def load_data_from_jsonl(file_path: str) -> list[dict]:
    """从jsonl文件中加载完整数据集。"""
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
    except FileNotFoundError:
        raise FileNotFoundError(f"文件未找到: {file_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"解析JSON时出错 {file_path}: {e}")
    return data


def load_single_question_data(file_path: str) -> dict:
    """从jsonl文件中加载单行（第一个）数据。"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            first_line = f.readline()
            if not first_line:
                raise ValueError(f"文件为空: {file_path}")
            return json.loads(first_line)
    except FileNotFoundError:
        raise FileNotFoundError(f"文件未找到: {file_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"解析JSON时出错 {file_path}: {e}")

# --- 用于评估的函数 ---
ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"

def extract_answer(completion: str) -> str:
    """从模型输出中提取数值答案。"""
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip().replace(",", "")
        return match_str
    else:
        lines = completion.strip().split('\n')
        last_line = lines[-1]
        cleaned_line = re.sub(r'[^\d\.\-]', '', last_line)
        try:
            float(cleaned_line)
            return cleaned_line
        except ValueError:
            return INVALID_ANS

def is_correct(model_completion: str, gt_example: dict) -> bool:
    """判断模型答案是否正确。"""
    gt_answer = extract_answer(gt_example["answer"])
    assert gt_answer != INVALID_ANS, f"标准答案格式错误！\n示例: {gt_example['answer']}"
    model_answer = extract_answer(model_completion)
    return model_answer == gt_answer