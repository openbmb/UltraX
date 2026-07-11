<p align="center">
  <a href="https://huggingface.co/collections/openbmb/ultradata">
    <img src="./assets/ultradata-logo.png" alt="OpenBMB UltraData" width="350"/>
  </a>
</p>

<h1 align="center">UltraX：基于自适应程序化编辑的大规模预训练数据精炼</h1>

<p align="center">
<a href="https://arxiv.org/abs/2607.08646">📜 论文</a> |
<a href="https://huggingface.co/datasets/openbmb/UltraX-Preview">🤗 数据集</a> |
<a href="https://huggingface.co/openbmb/UltraX-0.6B-Preview">🤗 模型</a> |
<a href="https://huggingface.co/collections/openbmb/ultradata">📦 UltraData 合集</a> |
<a href="https://opensource.org/license/apache-2-0">📄 License: Apache 2.0</a>
</p>

<p align="center">
<a href="README.md">English</a> |
中文
</p>

## 📚 简介

**UltraX** 是一个面向大规模预训练数据的函数调用式精炼框架，可为每条样本自适应生成并执行编辑函数，实现高效的逐例精炼。与端到端文本改写不同，UltraX 训练一个轻量精炼模型来预测结构化编辑操作——包括插入、删除和修改——然后在原始文本上确定性执行。

![UltraX Pipeline](./assets/ultrax_pipeline.png)

**核心特性：**

- **完整函数空间：** 覆盖插入（`add_line`）、删除（`remove_lines`、`remove_all`）和修改（`replace_str`）操作，实现超越仅删除方法的细粒度实例级编辑。
- **可靠的种子监督：** 利用专家 LLM 端到端精炼 + LAM（行级对齐与映射）+ DCR（动态上下文替换），生成高质量的结构化程序监督。
- **鲁棒的大规模执行：** 滑动窗口推理 + 重叠感知操作聚合、歧义过滤、同行操作合并和重复模式检测，确保大规模执行的可靠性。
- **最优性能：** 在 1B 模型从零预训练实验中，UltraX 在所有五个语料上均取得最高平均性能，多个数据集性能提升超过 2%。

## 📢 最新动态

- **[2026.07]** 🎉 UltraX 代码、精炼模型和精炼数据集正式开源。

## 💡 亮点

- **函数调用式精炼：** UltraX 不进行端到端文本改写，而是预测结构化编辑操作（`keep_all`、`remove_all`、`remove_lines`、`replace_str`、`add_line`），实现细粒度实例级编辑与确定性执行。
- **LAM + DCR 流水线：** 行级对齐与映射（LAM）在行级别对齐原始文本与精炼文本，动态上下文替换（DCR）将字符级编辑转化为具有唯一上下文锚定的可靠 `replace_str` 操作。
- **鲁棒的大规模执行：** 滑动窗口推理 + 重叠感知操作聚合、歧义过滤、同行操作合并和重复模式检测，确保大规模执行的可靠性。
- **最优性能：** 在 1B 模型从零预训练实验中，UltraX 在所有五个语料上均取得最高平均性能，多个数据集性能提升超过 2%，并展现出更高的数据效率。

## 📋 流程概览

UltraX 流程包含两个主要阶段：

### 阶段一：精炼模型构建

| 步骤 | 模块 | 说明 |
|------|------|------|
| 1 | `seed_preprocessing/` | 长文档滑动窗口切分（12K token 窗口，20% 重叠） |
| 2 | `prompt_optimization/` | 数据集自适应 prompt 优化（画像 + 迭代精炼 + 回归测试） |
| 3 | `e2e_refinement/` | 通过专家 LLM API 进行批量端到端文本精炼 |
| 4 | `function_construction/` | LAM + DCR：将（原文, 精炼文本）对转化为结构化函数调用训练数据 |
| 5 | `sft_data_building/` | 编辑偏置采样 + 系统指令注入 |
| 6 | `model_training/` | 使用 ms-swift + DeepSpeed ZeRO3 进行全参数 SFT |

### 阶段二：大规模程序执行

| 步骤 | 模块 | 说明 |
|------|------|------|
| 7 | `inference/` | 多 GPU 数据并行 vLLM 推理 + 滑动窗口分段 |
| 8 | `post_processing/` | 函数解析、校验、后处理与确定性执行 |

## 📈 评估结果

使用 1B 参数的 MiniCPM 模型在 20B tokens 上从零预训练，10 个基准零样本评估。

<div align="center">
  <img src="./assets/results.png" width="900"/>
</div>

UltraX 在**所有五个语料上均取得最高平均性能**，在 50 个任务-语料组合中赢得 34 个最佳结果。

<div align="center">
  <img src="./assets/fineweb_token_curve.png" alt="FineWeb Token 曲线" width="450"/>
  <p><i>FineWeb 在不同训练 token 预算下的平均下游性能。</i></p>
</div>

## 🚀 快速开始

### 环境配置

```bash
git clone https://github.com/BIGWangYuDong/UltraX.git
cd UltraX
conda create -n ultrax python=3.10
conda activate ultrax
pip install -r requirements.txt
```

推理还需要安装 vLLM：

```bash
pip install vllm
```

模型训练需要安装 ms-swift：

```bash
pip install ms-swift
```

### 阶段一：精炼模型构建

#### 步骤 1：种子数据预处理

将长文档按行边界切分为重叠窗口：

```bash
python stage1_model_construction/seed_preprocessing/sliding_window_splitter.py \
    --input_dir /path/to/seed_data \
    --output_dir /path/to/output \
    --tokenizer_path /path/to/tokenizer \
    --max_tokens 12000 \
    --overlap_ratio 0.2
```

#### 步骤 2：数据集自适应 Prompt 优化

为每个数据集自动优化精炼 prompt：

```bash
cd stage1_model_construction/prompt_optimization
python main.py \
    --api-key $API_KEY \
    --datasets fineweb redpajama \
    --max-iterations 200 \
    --batch-size 5
```

#### 步骤 3：端到端精炼

使用优化后的 prompt 通过专家 LLM 精炼种子数据：

```bash
python stage1_model_construction/e2e_refinement/refine_dataset.py \
    --input_dir /path/to/seed_data \
    --output_dir /path/to/refined_output \
    --prompt_dir /path/to/optimized_prompts \
    --api_url $API_URL \
    --api_key $API_KEY \
    --model deepseek-v3
```

#### 步骤 4：函数构造（LAM + DCR）

将（原文, 精炼文本）对转化为结构化函数调用训练数据：

```bash
python stage1_model_construction/function_construction/function_construction.py \
    --input_dir /path/to/refined_data \
    --output_dir /path/to/training_data \
    --num_workers 64
```

#### 步骤 5：SFT 数据构建

按操作组合类别采样训练数据并添加系统指令：

```bash
python stage1_model_construction/sft_data_building/sample_and_format.py \
    --train_data_dir /path/to/training_data \
    --output_dir /path/to/sft_data
```

#### 步骤 6：模型训练

使用 ms-swift 训练精炼模型：

```bash
bash stage1_model_construction/model_training/train.sh
```

> **注意：** 运行前请修改 `train.sh` 中的路径。详细超参数设置见脚本内容。

### 阶段二：大规模程序执行

#### 步骤 7：推理

运行多 GPU 数据并行 vLLM 推理：

```bash
python stage2_large_scale_execution/inference/inference.py \
    --input_dir /path/to/raw_data \
    --model_path /path/to/trained_model \
    --output_dir /path/to/inference_output \
    --num_gpus 8 \
    --max_chars 48000
```

#### 步骤 8：后处理与执行

解析、校验并执行预测的清洗操作：

```bash
python stage2_large_scale_execution/post_processing/post_process_and_execute.py \
    --input_dir /path/to/inference_output \
    --output_dir /path/to/cleaned_data \
    --num_workers 128
```

输出包含 3 列：`original`、`cleaned`、`processed_functions`。

## 🔧 函数空间

| 函数 | 说明 |
|------|------|
| `keep_all()` | 文档无需修改 |
| `remove_all()` | 整篇文档无价值（如错误页面、登录墙） |
| `remove_lines(start, end)` | 删除从 start 到 end 的连续行（含首尾） |
| `replace_str(line, old, new)` | 在指定行内替换子字符串 |
| `add_line(base, sub_idx, content)` | 在指定位置附近插入新行 |

## ❤️ 致谢

感谢以下项目的卓越贡献：

- [ms-swift](https://github.com/modelscope/ms-swift) — 模型训练框架
- [vLLM](https://github.com/vllm-project/vllm) — 高吞吐推理引擎
- [ProX](https://github.com/GAIR-NLP/ProX) — 程序化数据精炼先驱
- [LightEval](https://github.com/huggingface/lighteval) — 评估框架

## 📖 引用

如果我们的工作对您有帮助，请考虑引用：

```bibtex
@misc{ultrax2026,
  title={UltraX: Refining Pre-Training Data at Scale with Adaptive Programmatic Editing},
  author={Xinlong Zhao and Dongsheng Liu and Hengyu Zhao and Zixuan Fu and Zheng Wang and Jie Cai and Jie Zhou and Qiang Ma and Xuanhe Zhou and Xu Han and Yudong Wang and Zhiyuan Liu},
  year={2026},
  eprint={2607.08646},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
}
```

## 📜 许可证

本项目基于 [Apache 2.0](./LICENSE) 许可证发布。

**禁止不加处理的二次发布：** 未经原作者（或本机构）书面明确授权，任何其他机构、组织或第三方平台不得以任何形式对本项目成果进行直接转载、复制、托管、镜像克隆或商业化包装再发布。
