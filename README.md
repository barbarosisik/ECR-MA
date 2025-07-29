# ECR-MA: LLM-Based Empathetic Conversational Recommender Multi-Agent System

This repository contains the code for a modern, RL-enhanced Empathetic Conversational Recommender (ECR) system, with a focus on **LLM-based response evaluation, critic agent training, and reinforcement learning** for improved dialogue quality and recommendation realism.

## Features

- **LLM-Based Scoring:** Uses Llama2-Chat and Mistral7B-Instruct (for critic only; all generation is now Llama2-Chat).
- **Critic Agent:** Trains a RoBERTa-based critic agent to predict multiple quality metrics for dialogue responses.
- **RL Pipeline:** Integrates PPO-based reinforcement learning to optimize response generation using the critic as a reward signal.
- **SLURM/Cluster Ready:** Includes scripts for large-scale scoring and training on GPU clusters (ALICE/DS-Lab).
- **Comprehensive Evaluation:** Scripts for visualization, metric analysis, and benchmark comparison.

## Requirements

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

## Recent Major Change
- The system now uses **Llama2-Chat as the ONLY backbone** for response generation (no DialoGPT).
- All training, inference, and evaluation are aligned with the ECR paper's Llama2-Chat structure and prompt format.
- Data: Llama2-annotated emotional data (`llama_train.json`, `llama_test.json`).
- Training: Fine-tune Llama2-Chat for emotion-aligned response generation using emotional reviews and knowledge-augmented prompts.
- Evaluation: Use subjective metrics (emotional intensity, persuasiveness, informativeness, lifelikeness) and LLM-based scoring if possible.

## Quick Start

### 1. LLM-Based Scoring

**Llama2-Chat (Completed):**
```bash
python convert_and_score_full_dataset.py --input <input_file> --output <output_file>
```

**Mistral7B-Instruct:**
```bash
python src_emo/scoring/mistral7b_score_responses_ultra_fast.py --input <input_file> --output <output_file>
```

**Note:** Mixtral8x7B-Instruct was attempted but requires GPU memory beyond available resources (46.7B parameters vs 10.57 GiB available).

## Recent Fixes
- Fixed RL model saving issue (tensor sharing error) by using `safe_serialization=False`.
- Lowered RL checkpoint save frequency to every 100 steps for better checkpointing.
- Mixtral8x7B and DialoGPT-large were abandoned due to technical and hardware issues.

## Immediate Next Steps
1. ✅ **ECR Model Evaluation**: Completed comprehensive evaluation of LoRA-enhanced Llama2 model
2. 🔄 **RL Training**: Currently running optimized RL training (Job 4512034) - 20K samples, 2 epochs, 24h limit
3. Test the RL-enhanced model with a test/inference script.
4. Update all documentation files to reflect the new status.

## Current Status

### Completed
- **Llama2-70B-Chat scoring**: All 6 parts completed successfully
- **Mistral7B-Instruct scoring**: All 6 parts completed successfully  
- **Total scored samples**: 16,716 (2,786 per part × 6 parts)
- **Critic agent trained on merged dual-model data**
- **RL training pipeline implemented and debugged**
- **ECR Model Evaluation**: Comprehensive evaluation of LoRA-enhanced Llama2 model completed
  - Fixed model loading issues (LoRA adapter integration)
  - Implemented proper conversation formatting for Llama2
  - Achieved good diversity scores (Distinct-1: 0.78, Distinct-2: 0.95)
  - Generated meaningful responses (avg 73.2 words)
  - Identified areas for improvement in BLEU scoring

### Next Steps
1. **Retrain RL** with fixed model saving and improved checkpointing
2. **Test RL-enhanced model**
3. **Update documentation and prepare final results**

## Model Selection Rationale

Based on the MACRS paper analysis, we selected the same LLM backbones they used:

- **Llama2-70B-Chat**: Open-source model with 70B parameters, used in MACRS-L variant
- **Mistral7B-Instruct**: High-performance 7B parameter model, similar to ChatGPT performance
- **Mixtral8x7B-Instruct**: Attempted but not feasible due to hardware constraints

The MACRS paper shows that these models provide excellent performance for conversational recommendation scoring, with MACRS-C (ChatGPT-based) achieving 61% success rate and MACRS-L (Llama2-based) achieving 48% success rate.

- See `score_*.slurm` files for cluster job submission.

### 2. Critic Training

Train the critic agent on dual-model scored data:
```bash
python src_emo/train_critic_supervised.py --train_data <train.jsonl> --val_data <val.jsonl> --output_dir critic_pretrained
```

### 3. RL Training

Run RL-enhanced training with PPO (Llama2-Chat backbone):
```bash
./run_rl_training.sh
# or manually:
accelerate launch src_emo/train_emp_rl.py --dataset redial --use_rl --critic_pretrained_path critic_pretrained/critic_pretrained_final.pth --output_dir models/rl_enhanced_ecr --model llama2-chat
```

### 4. Evaluation

Evaluate models and visualize results:
```bash
python eval_critic_visualize.py
python summarize_critic_evaluation.py
```

## Project Structure

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

## Documentation

- See `IMPLEMENTATION_GUIDE.md` and `RL_TRAINING_GUIDE.md` for detailed instructions.
- `CRITIC_EVALUATION_REPORT.md` and `PROGRESS_REPORT.md` for project progress and evaluation results.

## Acknowledgements

- Built on top of [UniCRS](https://github.com/RUCAIBox/UniCRS) and the original ECR-main codebase.
- LLM scoring inspired by recent RecSys and NLP research.
- For academic use, please cite the original ECR paper and this repository.

---

**For questions or collaboration, open an issue or contact the maintainer.**
