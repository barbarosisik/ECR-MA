import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from src_emo.rl import CriticAgent, RLConfig
import os

# Paths
MODEL_DIR = './models/rl_enhanced_ecr_improved_2025-07-21-15-38-34'
CRITIC_PATH = './critic_pretrained_dual_model/critic_final.pth'

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load RL-enhanced model and tokenizer
print('Loading RL-enhanced model...')
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR).to(device)

# Load critic agent
print('Loading critic agent...')
rl_config = RLConfig(device=device)
critic_tokenizer = AutoTokenizer.from_pretrained(rl_config.critic_model_name, cache_dir="/data1/s3905993/cache/huggingface", local_files_only=True)
emotion_list = ['happy', 'sad', 'angry', 'neutral']
critic = CriticAgent.load_model(CRITIC_PATH, rl_config, critic_tokenizer)
critic.eval()

# Sample contexts
sample_contexts = [
    "I'm feeling a bit down today. Can you recommend a movie to cheer me up?",
    "I just watched Inception and loved it. Any similar recommendations?",
    "I'm looking for a family-friendly movie for the weekend.",
    "I had a rough day at work. Any suggestions for a relaxing film?"
]

# Generate and evaluate responses
for i, context in enumerate(sample_contexts):
    print(f"\n=== Sample {i+1} ===")
    print(f"Context: {context}")
    # Tokenize and generate response
    input_ids = tokenizer.encode(context, return_tensors='pt').to(device)
    with torch.no_grad():
        output = model.generate(input_ids, max_length=100, num_beams=5, do_sample=True, top_p=0.95, top_k=50, eos_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(output[0], skip_special_tokens=True)[len(context):].strip()
    print(f"Generated Response: {response}")
    # Critic evaluation
    critic_metrics = critic.evaluate_responses([context], [response])
    print(f"Critic Evaluation: {critic_metrics}") 