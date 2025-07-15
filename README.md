# ECR-MA: LLM-Based Empathetic Conversational Recommender Multi-Agent System

This repository contains the code for a modern, RL-enhanced Empathetic Conversational Recommender (ECR) system, with a focus on **LLM-based response evaluation, multi-head critic training, and reinforcement learning** for improved dialogue quality and recommendation realism.

## 🚀 Features

- **LLM-Based Scoring:** Uses Llama2-Chat to generate fine-grained quality scores for conversational responses (empathy, informativeness, recommendation, engagement, overall).
- **Multi-Head Critic:** Trains a RoBERTa-based critic to predict multiple quality metrics for dialogue responses.
- **RL Pipeline:** Integrates PPO-based reinforcement learning to optimize response generation using the critic as a reward signal.
- **SLURM/Cluster Ready:** Includes scripts for large-scale scoring and training on GPU clusters (ALICE/DS-Lab).
- **Comprehensive Evaluation:** Scripts for visualization, metric analysis, and benchmark comparison.

## 📦 Requirements

- Python >= 3.8
- PyTorch >= 1.8
- CUDA Toolkit >= 11.1
- transformers >= 4.15
- accelerate >= 0.8
- wandb, nltk, pyg, tqdm

Install dependencies:
```bash
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt')"
```

## 📁 Data & Models

**Note:**  
- Data and model weights are **not included** in this repository due to size.
- Download the processed data and model checkpoints from:
  - [Redial/Emo Data](https://drive.google.com/file/d/1fb9kDo8uSRLlwc5c4nUw8DZHR5XOY_l_/view?usp=sharing)
  - [Model Checkpoints](https://drive.google.com/file/d/1uBtcqbQByVrrJ1hEwk2dvsAOxuvEgE19/view?usp=sharing)
- Place data in `src_emo/data/emo_data/` and models in `src_emo/data/saved/`.

## ⚡ Quick Start

### 1. LLM-Based Scoring

Run Llama2-based scoring on your dataset:
```bash
python convert_and_score_full_dataset.py --input <input_file> --output <output_file>
```
- See `score_full_dataset.slurm` for cluster job submission.

### 2. Critic Training

Train the multi-head critic on scored data:
```bash
python src_emo/train_critic_supervised.py --train_data <train.jsonl> --val_data <val.jsonl> --output_dir critic_pretrained
```

### 3. RL Training

Run RL-enhanced training with PPO:
```bash
./run_rl_training.sh
# or manually:
accelerate launch src_emo/train_emp_rl.py --dataset redial --use_rl --critic_pretrained_path critic_pretrained/critic_pretrained_final.pth --output_dir models/rl_enhanced_ecr
```

### 4. Evaluation

Evaluate models and visualize results:
```bash
python eval_critic_visualize.py
python summarize_critic_evaluation.py
```

## 📝 Project Structure

```
ECR-main/
├── src_emo/                  # Main source code (RL, critic, data processing)
├── convert_and_score_full_dataset.py
├── create_quality_labels.py
├── eval_critic_visualize.py
├── summarize_critic_evaluation.py
├── run_rl_training.sh
├── run_rl_evaluation.sh
├── score_full_dataset.slurm
├── README.md
└── ... (see .gitignore for excluded data/models)
```

## 📊 Documentation

- See `IMPLEMENTATION_GUIDE.md` and `RL_TRAINING_GUIDE.md` for detailed instructions.
- `CRITIC_EVALUATION_REPORT.md` and `PROGRESS_REPORT.md` for project progress and evaluation results.

## 📚 Acknowledgements

- Built on top of [UniCRS](https://github.com/RUCAIBox/UniCRS) and the original ECR-main codebase.
- LLM scoring inspired by recent RecSys and NLP research.
- For academic use, please cite the original ECR paper and this repository.

---

**For questions or collaboration, open an issue or contact the maintainer.**
