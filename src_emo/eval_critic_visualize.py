import os
import json
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from rl.critic import CriticAgent
from transformers import AutoTokenizer

MODEL_PATH = "../critic_pretrained_rich/best_critic_pretrained.pth"
DATA_PATH = "../llama2_scored_rich.jsonl"
MODEL_NAME = "roberta-base"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SCORE_NAMES = [
    'empathy_score',
    'informativeness_score',
    'recommendation_score',
    'engagement_score',
    'overall_score',
    'bleu_score',
]

# Load data
samples = []
with open(DATA_PATH) as f:
    for line in f:
        item = json.loads(line)
        samples.append(item)

# Load model
class CriticConfig:
    def __init__(self, model_name, device):
        self.critic_model_name = model_name
        self.critic_hidden_size = 768
        self.critic_dropout = 0.1
        self.device = device
    def to_dict(self):
        return self.__dict__
config = CriticConfig(MODEL_NAME, DEVICE)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
emotion_list = ["like", "curious", "happy", "grateful", "negative", "neutral", "nostalgia", "agreement", "surprise"]
critic = CriticAgent(config, tokenizer, emotion_list).to(DEVICE)
critic.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
critic.eval()

# Run inference
all_scores = {name: [] for name in SCORE_NAMES}
examples = []
with torch.no_grad():
    for item in tqdm(samples):
        context = item['context']
        response = item['response']
        out = critic([context], [response], return_quality_breakdown=True)
        scores = out['quality_breakdown']
        for name in SCORE_NAMES:
            all_scores[name].append(float(scores[name][0].cpu().numpy()))
        # Save a few examples
        if len(examples) < 5:
            examples.append({
                'context': context,
                'response': response,
                'scores': {name: float(scores[name][0].cpu().numpy()) for name in SCORE_NAMES}
            })

# Plot histograms
plt.figure(figsize=(12, 8))
for i, name in enumerate(SCORE_NAMES):
    plt.subplot(2, 3, i+1)
    plt.hist(all_scores[name], bins=10, alpha=0.7)
    plt.title(name)
plt.tight_layout()
plt.savefig("critic_score_distributions.png")
plt.show()

# Print example predictions
print("\nExample predictions:")
for ex in examples:
    print("Context:", ex['context'][:120], "...")
    print("Response:", ex['response'])
    print("Scores:", ex['scores'])
    print("-"*60) 