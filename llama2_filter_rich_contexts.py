import json
from tqdm import tqdm

INPUT_FILE = "llama2_prompts.jsonl"
OUTPUT_FILE = "llama2_prompts_rich.jsonl"
MIN_CONTEXT_TOKENS = 20

def count_tokens(text):
    return len(text.strip().split())

def main():
    kept = 0
    total = 0
    with open(INPUT_FILE) as fin, open(OUTPUT_FILE, "w") as fout:
        for line in tqdm(fin):
            total += 1
            item = json.loads(line)
            context = item.get("context", "")
            if count_tokens(context) >= MIN_CONTEXT_TOKENS:
                fout.write(json.dumps(item, ensure_ascii=False) + "\n")
                kept += 1
    print(f"Kept {kept} of {total} prompts with context >= {MIN_CONTEXT_TOKENS} tokens.")

if __name__ == "__main__":
    main() 