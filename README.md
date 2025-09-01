# 探索大语言模型Soft Prompt的拓扑结构

本项目旨在训练和分析大型语言模型（LLM）的“软提示”（Soft Prompts）。我们不仅关注如何通过优化连续的嵌入向量来引导模型生成特定答案，更深入地利用**拓扑数据分析（Topological Data Analysis, TDA）** 技术，来探索这些软提示在训练过程中的几何与结构演化。

项目的核心思想是：软提示在训练过程中学习到的高维嵌入向量流形，其拓扑特征（如连通分支、环路等）的变化，可能与模型的学习状态和最终性能相关。

## 目录

  - [主要功能](#主要功能)
  - [项目结构](#项目结构)
  - [环境设置与安装](#环境设置与安装)
  - [使用指南](#使用指南)
      - [第一步：准备数据](#第一步准备数据)
      - [第二步：训练Soft Prompt](#第二步训练soft-prompt)
      - [第三步：评估模型准确率](#第三步评估模型准确率)
      - [第四步：执行TDA分析](#第四步执行tda分析)

## 主要功能

1.  **两种训练模式**：

      * **单个问答对训练 (`train_on_single_question.py`)**: 在单个问题上对Soft Prompt进行过拟合训练，便于快速调试和观察提示向量的收敛过程。
      * **完整数据集训练 (`train_on_dataset.py`)**: 在标准数据集（如BBH）上进行训练，模拟更真实的场景，训练出具有泛化能力的Soft Prompt。

2.  **性能追踪与评估 (`track_accuracy_over_epochs.py`)**:

      * 自动评估训练过程中保存的所有Soft Prompt检查点。
      * 生成详细的日志，记录每个epoch的模型输出和正确率，便于分析模型性能随训练的演变。

3.  **拓扑数据分析 (`tda_analysis_of_softprompts.py`)**:

      * 使用`Gudhi`库对每个epoch保存的Soft Prompt嵌入向量进行持久同调分析。
      * 计算关键TDA指标：
          * $H\_0$ (零维同调群)：连通分支的数量。
          * $H\_1$ (一维同调群)：环路的数量。
          * 特征的生命周期（Lifetime）。
          * 持久性熵（Persistent Entropy）。
      * 可视化分析结果，包括持久性条带图（Barcode）、持久性图（Diagram）以及各项指标随训练epoch的变化曲线。

4.  **模块化代码 (`utils.py`)**:

      * 将模型加载、分词、数据处理、推理生成等核心共享功能重构到`utils.py`中，提高了代码的可读性、复用性和可维护性。

## 项目结构

```
.
├── data/
│   ├── single/
│   │   └── single_HotpotQA_data.jsonl      # 单问题训练数据示例
│   └── dataset/
│       └── BBH.jsonl                       # 数据集训练数据示例
├── models/                                 # 存放训练好的Soft Prompt检查点 (.pt文件)
├── results/                                # 存放训练过程中的推理结果和损失曲线图
├── tda_analysis/                           # 存放TDA分析生成的可视化图表
├── train_on_single_question.py             # 脚本：在单个问题上训练
├── train_on_dataset.py                     # 脚本：在完整数据集上训练
├── track_accuracy_over_epochs.py           # 脚本：评估所有检查点的准确率
├── tda_analysis_of_softprompts.py          # 脚本：对Soft Prompt进行TDA分析
├── utils.py                                # 核心工具函数模块
├── requirements.txt                        # Python依赖库
└── README.md                               # 本文档
```

## 环境设置与安装

1.  **先决条件**:

      * Python 3.9+
      * NVIDIA GPU 以及正确安装的 CUDA Toolkit (推荐)
      * Git

2.  **克隆仓库**:

    ```bash
    git clone https://github.com/zhang-song-bo/Analyze-softprompt-using-TDA
    cd Analyze-softprompt-using-TDA
    ```

3.  **创建虚拟环境 (推荐)**:

    ```bash
    python -m venv venv
    # Windows
    .\venv\Scripts\activate
    # macOS / Linux
    source venv/bin/activate
    ```

4.  **安装依赖**:
    项目所需的所有依赖项都已在 `requirements.txt` 中列出。运行以下命令进行安装：

    ```bash
    pip install -r requirements.txt
    ```

## 使用指南

请按照以下步骤运行完整的训练和分析流程。

### 第一步：准备数据

  * **数据格式**: 所有数据都应为`.jsonl`格式，每一行是一个JSON对象，至少包含`"question"`和`"answer"`两个字段。
  * **存放位置**:
      * 用于单问题训练的数据，请放入 `data/single/` 目录。
      * 用于完整数据集训练的数据，请放入 `data/dataset/` 目录。

### 第二步：训练Soft Prompt

您可以根据需求选择以下任一训练模式。

  * **模式A: 在单个问题上训练**

    此脚本使用`argparse`来接收命令行参数，方便调整。

    ```bash
    python train_on_single_question.py \
        --data_path "<your data path>" \
        --model_save_dir "../models/" \
        --results_save_dir "../results/" \
        --model_name "google/gemma-2b-it" \
        --epochs 300 \
        --save_every 2 \
        --lr 5e-6
    ```

      * **输出**:
          * Soft Prompt检查点 (`.pt`文件) 会保存在 `--model_save_dir` 指定的目录中。
          * 每个检查点的推理结果和最终的损失曲线图会保存在 `--results_save_dir` 目录中。

  * **模式B: 在完整数据集上训练**

    此脚本使用`argparse`来接收命令行参数，方便调整。

    ```bash
    python train_on_dataset.py \
        --dataset_path "<your data path>" \
        --model_save_dir "../models_bbh/" \
        --results_save_dir "../results_bbh/" \
        --model_name "google/gemma-2b-it" \
        --epochs 200 \
        --save_every 5 \
        --batch_size 4 \
        --accumulation_steps 4 \
        --lr 2e-5    
    ```

      * **输出**:
          * 与模式A类似，检查点保存在 `models/` 目录，推理结果和损失曲线图保存在 `results/` 目录。

### 第三步：评估模型准确率

训练完成后，您可以运行此脚本来评估所有保存的检查点在特定问题上的表现。

```bash
python track_accuracy_over_epochs.py \
    --prompt_dir "../models/" \
    --data_path "<your data path>" \
    --output_log_path "../results/full_evaluation_log.txt" \
    --model_name "google/gemma-2b-it"
```

  * **输出**:
      * 一个详细的日志文件 (`full_evaluation_log.txt`) 将被创建，其中包含每个检查点的模型输出、正确性判断以及最终的总体准确率。

### 第四步：执行TDA分析

这是项目的核心分析步骤。此脚本将加载第二步中训练好的所有Soft Prompt检查点，并对它们进行拓扑数据分析。

**注意**: 请确保脚本中的 `MODEL_SAVE_DIR` 和 `RESULTS_SAVE_DIR` 路径正确指向您的模型和结果目录。

```bash
python tda_analysis_of_softprompts.py
```

  * **输出**:
      * 所有分析图表将保存在 `tda_analysis/` 目录中，包括：
          * 部分epoch的持久性条带图和持久性图。
          * $H\_0$/$H\_1$ 特征数量随epoch变化的曲线图。
          * $H\_0$/$H\_1$ 平均生命周期随epoch变化的曲线图。
          * 持久性熵随epoch变化的曲线图。

-----