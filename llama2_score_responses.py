import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm import tqdm
import argparse

MODEL_NAME = "/data1/s3905993/ECRHMAS/src/models/llama2_chat"  # Local path to Llama2-Chat model
PROMPT_FILE = "llama2_prompts.jsonl"
OUTPUT_FILE = "llama2_scored.jsonl"

def get_llama2_scores(prompt, model, tokenizer):
    # Force the model to return only JSON
    system_message = (
        "DO NOT SAY ANYTHING EXCEPT THE JSON OBJECT. "
        "Your output MUST be a valid JSON object with these five fields: "
        "Empathy, Informativeness, Recommendation quality, Engagement, Overall quality. "
        "No explanation, no extra text."
    )
    full_prompt = system_message + "\n" + prompt
    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
    output = model.generate(**inputs, max_new_tokens=256)
    result = tokenizer.decode(output[0], skip_special_tokens=True)
    try:
        json_start = result.find('{')
        json_end = result.rfind('}') + 1
        scores = json.loads(result[json_start:json_end])
        return scores
    except Exception as e:
        print("Failed to parse Llama2 output:", result)
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max', type=int, default=None, help='Max number of prompts to process (for testing)')
    args = parser.parse_args()

    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto")
    print("Model loaded.")

    with open(PROMPT_FILE) as fin, open(OUTPUT_FILE, "w") as fout:
        for i, line in enumerate(tqdm(fin)):
            if args.max is not None and i >= args.max:
                break
            item = json.loads(line)
            prompt = item["llama2_prompt"]
            scores = get_llama2_scores(prompt, model, tokenizer)
            if scores:
                item["llama2_scores"] = scores
                fout.write(json.dumps(item, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main() 