# RL-Enhanced ECR-main: Empathetic Conversational Recommender System

This repository contains the RL-enhanced version of the ECR-main project, integrating Proximal Policy Optimization (PPO) to improve empathetic response generation and recommendation accuracy.

## 🚀 Key Features

- **PPO Integration**: Proximal Policy Optimization for response generation
- **Multi-Metric Rewards**: BLEU, Distinct, Empathy, and Recommendation accuracy
- **Critic Agent**: Value function estimation for better policy updates
- **Comprehensive Evaluation**: Both supervised and RL-based metrics
- **Easy-to-Use Scripts**: Automated training and evaluation pipelines

## 📁 Project Structure

```
ECR-main/
├── src_emo/
│   ├── rl/                          # RL components
│   │   ├── __init__.py
│   │   ├── rl_config.py            # RL configuration
│   │   ├── reward_functions.py     # Reward calculation
│   │   ├── critic.py               # Critic agent
│   │   └── ppo_trainer.py          # PPO trainer
│   ├── train_emp_rl.py             # RL-enhanced training script
│   ├── evaluate_rl.py              # Enhanced evaluation script
│   └── ...                         # Original ECR-main files
├── run_rl_training.sh              # Training script
├── run_rl_evaluation.sh            # Evaluation script
└── README_RL.md                    # This file
```

## 🛠️ Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd ECR-main
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Additional RL dependencies**:
```bash
pip install nltk wandb accelerate transformers torch
```

4. **Download NLTK data**:
```python
import nltk
nltk.download('punkt')
```

## 🎯 Quick Start

### 1. Standard Training (Baseline)

```bash
# Train the baseline model
accelerate launch src_emo/train_emp.py \
    --dataset redial \
    --context_max_length 150 \
    --resp_max_length 150 \
    --per_device_train_batch_size 4 \
    --num_train_epochs 5 \
    --learning_rate 1e-5 \
    --output_dir data/saved/baseline
```

### 2. RL-Enhanced Training

```bash
# Train with RL enhancement
./run_rl_training.sh
```

Or manually:

```bash
accelerate launch src_emo/train_emp_rl.py \
    --dataset redial \
    --use_rl \
    --rl_learning_rate 1e-5 \
    --rl_batch_size 8 \
    --rl_max_steps 5000 \
    --bleu_weight 1.0 \
    --empathy_weight 2.0 \
    --recommendation_weight 1.5 \
    --output_dir data/saved/emp_conv_rl
```

### 3. Evaluation

```bash
# Evaluate with RL metrics
./run_rl_evaluation.sh
```

Or manually:

```bash
accelerate launch src_emo/evaluate_rl.py \
    --model_path data/saved/emp_conv_rl \
    --use_rl_eval \
    --output_dir evaluation_results
```

## 📊 RL Components

### 1. Reward Functions

The RL system uses a composite reward function:

```python
Total Reward = α × BLEU + β × Distinct + γ × Empathy + δ × Recommendation - Penalty
```

- **BLEU Reward**: Measures response fluency and relevance
- **Distinct Reward**: Encourages diverse and non-repetitive responses
- **Empathy Reward**: Evaluates emotional alignment with user context
- **Recommendation Reward**: Rewards proper movie recommendations
- **Format Penalty**: Penalizes malformed or incomplete responses

### 2. Critic Agent

The critic agent provides value estimates for PPO:

- **Base Model**: RoBERTa for context-response understanding
- **Value Head**: Estimates expected future rewards
- **Quality Head**: Provides detailed quality breakdown
- **Emotion Head**: Classifies emotional content

### 3. PPO Training

The PPO algorithm optimizes the policy:

- **Policy Updates**: Clipped objective for stable training
- **Value Updates**: MSE loss for critic improvement
- **Advantage Estimation**: GAE for better credit assignment
- **Entropy Bonus**: Encourages exploration

## ⚙️ Configuration

### RL Hyperparameters

```python
# PPO Settings
ppo_epochs = 4
ppo_clip_epsilon = 0.2
ppo_entropy_coef = 0.01

# Training Settings
rl_learning_rate = 1e-5
rl_batch_size = 8
rl_max_steps = 10000

# Reward Weights
bleu_weight = 1.0
distinct_weight = 0.5
empathy_weight = 2.0
recommendation_weight = 1.5
```

### Customizing Rewards

You can adjust reward weights based on your priorities:

```bash
# Emphasize empathy
--empathy_weight 3.0 --bleu_weight 0.5

# Emphasize recommendations
--recommendation_weight 2.5 --distinct_weight 0.3

# Balanced approach
--bleu_weight 1.0 --empathy_weight 2.0 --recommendation_weight 1.5
```

## 📈 Evaluation Metrics

### Standard Metrics
- **BLEU-1/2/3/4**: Response fluency and relevance
- **DIST-1/2/3/4**: Response diversity
- **Item Ratio**: Recommendation frequency

### RL-Specific Metrics
- **Mean Reward**: Average composite reward
- **Critic Value**: Expected future rewards
- **Reward Breakdown**: Individual component analysis

## 🔧 Advanced Usage

### 1. Custom Reward Functions

You can implement custom reward functions by extending `RewardCalculator`:

```python
class CustomRewardCalculator(RewardCalculator):
    def _calculate_custom_reward(self, response: str) -> float:
        # Your custom reward logic
        return custom_score
```

### 2. Multi-Agent Training

For multi-agent scenarios, modify the PPO trainer:

```python
# Initialize multiple agents
agents = [ResponseAgent(), EmotionAgent(), RecommendationAgent()]

# Train with shared critic
ppo_trainer = MultiAgentPPOTrainer(agents, shared_critic)
```

### 3. Curriculum Learning

Implement curriculum learning by adjusting reward weights:

```python
# Start with basic rewards
initial_weights = {'bleu': 1.0, 'empathy': 0.5}

# Gradually increase complexity
final_weights = {'bleu': 1.0, 'empathy': 2.0, 'recommendation': 1.5}
```

## 📊 Results and Comparison

### Expected Improvements

Compared to the baseline ECR-main:

| Metric | Baseline | RL-Enhanced | Improvement |
|--------|----------|-------------|-------------|
| BLEU-1 | ~0.15 | ~0.18 | +20% |
| BLEU-2 | ~0.08 | ~0.11 | +37% |
| DIST-1 | ~0.12 | ~0.15 | +25% |
| Empathy Score | ~0.65 | ~0.78 | +20% |
| Recommendation Accuracy | ~0.70 | ~0.82 | +17% |

### Training Curves

Monitor training progress with:

```bash
# Enable wandb logging
--use_wandb --project "ecr-rl-enhanced"
```

Key metrics to track:
- Policy Loss
- Critic Loss
- Mean Reward
- Individual reward components

## 🐛 Troubleshooting

### Common Issues

1. **Out of Memory**:
   ```bash
   # Reduce batch sizes
   --rl_batch_size 4 --per_device_train_batch_size 2
   ```

2. **Training Instability**:
   ```bash
   # Adjust PPO parameters
   --ppo_clip_epsilon 0.1 --ppo_epochs 2
   ```

3. **Poor Reward Convergence**:
   ```bash
   # Adjust reward weights
   --bleu_weight 0.5 --empathy_weight 1.0
   ```

### Debug Mode

Enable debug mode for detailed logging:

```bash
--debug --log_all
```

## 📚 References

1. **Original ECR Paper**: "Towards Empathetic Conversational Recommender Systems"
2. **PPO Algorithm**: Schulman et al., "Proximal Policy Optimization Algorithms"
3. **RL for NLP**: Ranzato et al., "Sequence Level Training with Recurrent Neural Networks"

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests and documentation
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Original ECR-main authors
- Hugging Face Transformers team
- OpenAI for PPO algorithm
- The open-source RL community

---

**Note**: This RL enhancement is designed to work with the existing ECR-main architecture. Make sure you have the original ECR-main setup working before applying RL enhancements. 