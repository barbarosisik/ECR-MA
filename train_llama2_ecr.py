import json
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
import torch
from torch.utils.data import Dataset
from peft import LoraConfig, get_peft_model

# Paths
MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"  # Change to your local Llama2-Chat path if needed
TRAIN_FILE = "src_emo/data/emo_data/llama_train.json"
OUTPUT_DIR = "models/llama2_ecr_finetuned"
CACHE_DIR = "/data1/s3905993/cache/huggingface"

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Custom Dataset
class LlamaECRDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=256):
        self.samples = []
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item in data:
                prompt = item['input'].strip() + '\n' + item['instruction'].strip()
                target = item['output'].strip()
                full_text = prompt + '\n' + target
                enc = tokenizer(full_text, max_length=max_length, truncation=True, padding='max_length')
                enc['labels'] = enc['input_ids'].copy()
                self.samples.append(enc)
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        return self.samples[idx]

# Load tokenizer and model
print('Loading Llama2-Chat model and tokenizer...')
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR, local_files_only=True)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR, local_files_only=True).to(device)

# LoRA/PEFT setup
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Load dataset
print('Loading training data...')
dataset = LlamaECRDataset(TRAIN_FILE, tokenizer)

# Data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Training arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,
    num_train_epochs=2,
    per_device_train_batch_size=16,  # LoRA allows larger batch size
    save_steps=500,
    save_total_limit=2,
    logging_steps=50,
    learning_rate=5e-5,
    fp16=torch.cuda.is_available(),
    report_to=[],
)

# Trainer
print('Starting training...')
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

trainer.train()

# Save final LoRA adapters and tokenizer
print(f"Saving LoRA adapters and tokenizer to {OUTPUT_DIR} ...")
model.save_pretrained(OUTPUT_DIR, safe_serialization=False)
tokenizer.save_pretrained(OUTPUT_DIR)
print('Training complete!') 