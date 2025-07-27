# RL Enhancement Implementation Guide

This guide will walk you through implementing and using the RL enhancement for ECR-main step by step.

## 🎯 Implementation Roadmap

### Phase 1: Setup and Preparation ✅
- [x] RL module structure created
- [x] Configuration system implemented
- [x] Reward functions defined
- [x] Critic agent implemented
- [x] PPO trainer created

### Phase 2: Integration and Testing 🔄
- [ ] Test RL components individually
- [ ] Integrate with existing ECR-main
- [ ] Validate reward functions
- [ ] Test training pipeline
- [ ] Confirm RL model and checkpoint saving after retraining (NEW)

### Phase 3: Training and Evaluation 📊
- [ ] Run baseline training
- [ ] Run RL-enhanced training (with fixed saving/checkpointing)
- [ ] Compare results
- [ ] Analyze improvements
- [ ] Test RL-enhanced model (NEW)

### Phase 4: Optimization and Deployment 🚀
- [ ] Hyperparameter tuning
- [ ] Model optimization
- [ ] Production deployment
- [ ] Documentation updates

## 🛠️ Step-by-Step Implementation

### Step 1: Environment Setup

1. **Navigate to ECR-main directory**:
```bash
cd ECR-main
```

2. **Install additional dependencies**:
```bash
pip install nltk wandb accelerate transformers torch bitsandbytes
python -c "import nltk; nltk.download('punkt')"
```

3. **Verify installation**:
```bash
python -c "from src_emo.rl import RLConfig, PPOTrainer, CriticAgent, RewardCalculator; print('RL components imported successfully')"
```

4. **Set up model scoring systems**:
```bash
# Llama2 scoring (completed)
python convert_and_score_full_dataset.py --input <input_file> --output <output_file>

# Mistral7B scoring
python src_emo/scoring/mistral7b_score_responses_ultra_fast.py --input <input_file> --output <output_file>

# Note: DialoGPT-large was tested but found unsuitable for structured scoring
# Mixtral8x7B-Instruct access is being requested for future use
```

### Step 2: Data Preparation

1. **Ensure your data is properly formatted**:
```bash
# Check if data exists
ls -la src_emo/data/redial/
```

2. **Process data if needed**:
```bash
cd src_emo
python data/redial/process.py
```

### Step 3: Baseline Training

1. **Train the baseline model first**:
```bash
accelerate launch src_emo/train_emp.py \
    --dataset redial \
    --context_max_length 150 \
    --resp_max_length 150 \
    --per_device_train_batch_size 4 \
    --num_train_epochs 3 \
    --learning_rate 1e-5 \
    --output_dir data/saved/baseline \
    --num_warmup_steps 1000
```

2. **Evaluate baseline**:
```bash
accelerate launch src_emo/infer_emp.py \
    --dataset redial \
    --split test \
    --per_device_eval_batch_size 8 \
    --context_max_length 150 \
    --resp_max_length 150
```

### Step 4: RL Component Testing

1. **Test reward functions**:
```python
# Create test script: test_rewards.py
from src_emo.rl import RLConfig, RewardCalculator
from transformers import AutoTokenizer

config = RLConfig()
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
reward_calc = RewardCalculator(config, tokenizer, ["happy", "sad", "angry"])

# Test reward calculation
context = ["I'm feeling sad today"]
responses = ["I understand how you feel. Let me recommend a happy movie to cheer you up."]
targets = ["I'm sorry you're feeling down. Here's a great comedy: <movie>."]

rewards = reward_calc.calculate_reward(context, responses, targets)
print(f"Rewards: {rewards}")
```

2. **Test critic agent**:
```python
# Create test script: test_critic.py
from src_emo.rl import RLConfig, CriticAgent
from transformers import AutoTokenizer

config = RLConfig()
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
critic = CriticAgent(config, tokenizer, ["happy", "sad", "angry"])

# Test critic evaluation
context = ["I'm looking for a good movie"]
responses = ["I'd recommend <movie>. It's a great film!"]

outputs = critic(context, responses, return_quality_breakdown=True)
print(f"Values: {outputs['values']}")
print(f"Quality: {outputs['quality_breakdown']}")
```

### Step 5: RL Training

1. **Start with small-scale RL training**:
```bash
accelerate launch src_emo/train_emp_rl.py \
    --dataset redial \
    --use_rl \
    --rl_learning_rate 1e-5 \
    --rl_batch_size 4 \
    --rl_max_steps 1000 \
    --bleu_weight 1.0 \
    --empathy_weight 2.0 \
    --recommendation_weight 1.5 \
    --output_dir data/saved/emp_conv_rl_test \
    --debug
```

2. **Monitor training progress**:
```bash
# Check logs
tail -f log/*.log

# If using wandb
wandb login
# Then add --use_wandb to training command
```

### Step 6: Evaluation and Comparison

1. **Evaluate RL-enhanced model**:
```bash
accelerate launch src_emo/evaluate_rl.py \
    --model_path data/saved/emp_conv_rl_test \
    --use_rl_eval \
    --output_dir evaluation_results \
    --split test
```

2. **Compare results**:
```python
# Create comparison script: compare_results.py
import json

# Load baseline results
with open('baseline_results.json', 'r') as f:
    baseline = json.load(f)

# Load RL results
with open('evaluation_results/evaluation_results_*.json', 'r') as f:
    rl_results = json.load(f)

# Compare metrics
metrics = ['bleu@1', 'bleu@2', 'dist@1', 'dist@2', 'item_ratio']
for metric in metrics:
    if metric in baseline and metric in rl_results:
        improvement = (rl_results[metric] - baseline[metric]) / baseline[metric] * 100
        print(f"{metric}: {baseline[metric]:.4f} -> {rl_results[metric]:.4f} ({improvement:+.1f}%)")
```

## 🔧 Configuration Tuning

### Reward Weight Tuning

1. **Empathy-focused training**:
```bash
--empathy_weight 3.0 --bleu_weight 0.5 --distinct_weight 0.3
```

2. **Recommendation-focused training**:
```bash
--recommendation_weight 2.5 --bleu_weight 1.0 --empathy_weight 1.0
```

3. **Balanced approach**:
```bash
--bleu_weight 1.0 --empathy_weight 2.0 --recommendation_weight 1.5 --distinct_weight 0.5
```

### PPO Hyperparameter Tuning

1. **Conservative training**:
```bash
--ppo_clip_epsilon 0.1 --ppo_epochs 2 --rl_learning_rate 5e-6
```

2. **Aggressive training**:
```bash
--ppo_clip_epsilon 0.3 --ppo_epochs 6 --rl_learning_rate 2e-5
```

## 📊 Monitoring and Debugging

### Training Monitoring

1. **Key metrics to track**:
- Policy Loss (should decrease)
- Critic Loss (should decrease)
- Mean Reward (should increase)
- Individual reward components

2. **Warning signs**:
- Policy loss increasing rapidly
- Critic loss not converging
- Rewards not improving
- NaN values in losses

### Debugging Tips

1. **Enable debug mode**:
```bash
--debug --log_all
```

2. **Check reward breakdown**:
```python
# In your training script
reward_breakdown = reward_calculator.get_reward_breakdown(
    context[0], responses[0], targets[0]
)
print(f"Reward breakdown: {reward_breakdown}")
```

3. **Validate critic predictions**:
```python
# Check if critic values are reasonable
critic_outputs = critic(context, responses)
print(f"Critic values: {critic_outputs['values']}")
print(f"Value range: {critic_outputs['values'].min():.3f} - {critic_outputs['values'].max():.3f}")
```

## 🚀 Advanced Features

### 1. Multi-Agent Training

To implement multi-agent training:

```python
# Create multiple specialized agents
response_agent = ResponseGenerationAgent()
emotion_agent = EmotionRecognitionAgent()
recommendation_agent = RecommendationAgent()

# Train with shared critic
ppo_trainer = MultiAgentPPOTrainer(
    agents=[response_agent, emotion_agent, recommendation_agent],
    shared_critic=critic
)
```

### 2. Curriculum Learning

Implement curriculum learning:

```python
# Start with simple rewards
initial_weights = {'bleu': 1.0, 'empathy': 0.5}

# Gradually increase complexity
for epoch in range(num_epochs):
    empathy_weight = 0.5 + (epoch / num_epochs) * 1.5
    recommendation_weight = (epoch / num_epochs) * 1.5
    
    # Update reward weights
    reward_calculator.update_weights({
        'empathy': empathy_weight,
        'recommendation': recommendation_weight
    })
```

### 3. Custom Reward Functions

Add custom reward functions:

```python
class CustomRewardCalculator(RewardCalculator):
    def _calculate_safety_reward(self, response: str) -> float:
        """Calculate safety reward based on content filtering"""
        unsafe_words = ['inappropriate', 'offensive', 'harmful']
        safety_score = 1.0
        for word in unsafe_words:
            if word in response.lower():
                safety_score -= 0.2
        return max(0.0, safety_score)
    
    def calculate_reward(self, context, responses, targets, emotion_labels=None):
        # Get base rewards
        base_rewards = super().calculate_reward(context, responses, targets, emotion_labels)
        
        # Add custom rewards
        safety_rewards = torch.tensor([
            self._calculate_safety_reward(resp) for resp in responses
        ], device=self.config.device)
        
        return base_rewards + 0.5 * safety_rewards
```

## 📈 Expected Results

### Performance Improvements

Based on similar RL implementations, expect:

| Metric | Baseline | RL-Enhanced | Improvement |
|--------|----------|-------------|-------------|
| BLEU-1 | 0.15 | 0.18 | +20% |
| BLEU-2 | 0.08 | 0.11 | +37% |
| DIST-1 | 0.12 | 0.15 | +25% |
| Empathy | 0.65 | 0.78 | +20% |
| Recommendations | 0.70 | 0.82 | +17% |

### Training Time

- **Baseline training**: ~2-3 hours
- **RL training**: ~4-6 hours (additional time for reward computation)
- **Evaluation**: ~30 minutes

## 🐛 Common Issues and Solutions

### Issue 1: Out of Memory
**Solution**: Reduce batch sizes
```bash
--rl_batch_size 2 --per_device_train_batch_size 1
```

### Issue 2: Training Instability
**Solution**: Adjust PPO parameters
```bash
--ppo_clip_epsilon 0.1 --ppo_epochs 2 --rl_learning_rate 5e-6
```

### Issue 3: Poor Reward Convergence
**Solution**: Check reward weights and adjust
```bash
--bleu_weight 0.5 --empathy_weight 1.0 --recommendation_weight 0.5
```

### Issue 4: Critic Not Learning
**Solution**: Increase critic learning rate
```bash
--critic_learning_rate 2e-5
```

## 📚 Next Steps

1. **Immediate Actions**:
   - Test the RL components individually
   - Run a small-scale training experiment
   - Compare with baseline results

2. **Short-term Goals**:
   - Optimize hyperparameters
   - Implement multi-agent training
   - Add curriculum learning

3. **Long-term Goals**:
   - Deploy to production
   - Implement online learning
   - Add human feedback integration

## 🤝 Getting Help

If you encounter issues:

1. **Check the logs**: Look for error messages in log files
2. **Enable debug mode**: Use `--debug` flag for detailed output
3. **Validate data**: Ensure your dataset is properly formatted
4. **Check dependencies**: Verify all packages are installed correctly

## 📞 Support

For additional support:
- Check the main README_RL.md
- Review the original ECR-main documentation
- Open an issue in the repository

---

**Good luck with your RL enhancement implementation!** 🚀 

## 🚨 Recent Fixes
- Fixed RL model saving issue (tensor sharing error) by using `safe_serialization=False`.
- Lowered RL checkpoint save frequency to every 100 steps for better checkpointing.
- Mixtral8x7B and DialoGPT-large were abandoned due to technical and hardware issues.

## 🚨 Recent Major Change
- The system now uses **Llama2-Chat as the ONLY backbone** for response generation (no DialoGPT).
- All training, inference, and evaluation are aligned with the ECR paper's Llama2-Chat structure and prompt format.
- Data: Llama2-annotated emotional data (`llama_train.json`, `llama_test.json`).

## ⏩ Immediate Next Steps
1. Retrain RL for at least 1 epoch to confirm model and checkpoint saving.
2. Test the RL-enhanced model with a test/inference script.
3. Update all documentation files to reflect the new status. 