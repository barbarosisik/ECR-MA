from transformers import AutoModelForCausalLM, AutoTokenizer
import os

local_dir = "/data1/s3905993/ECRHMAS/src/models/mistral7b_instruct"
model_id = "mistralai/Mistral-7B-Instruct-v0.2"

os.makedirs(local_dir, exist_ok=True)

print(f"Downloading model to {local_dir} ...")
model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir=local_dir)
tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=local_dir)
print("Download complete.")

