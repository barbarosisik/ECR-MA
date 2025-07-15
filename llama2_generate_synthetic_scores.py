import json
import random
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

INPUT_FILE = "llama2_prompts_rich.jsonl"
OUTPUT_FILE = "llama2_scored_rich.jsonl"

FIELDS = [
    "Empathy",
    "Informativeness",
    "Recommendation quality",
    "Engagement",
    "Overall quality",
    "BLEU"
]

def compute_bleu(reference, candidate):
    # Tokenize by whitespace for simplicity
    ref_tokens = reference.strip().split()
    cand_tokens = candidate.strip().split()
    if not ref_tokens or not cand_tokens:
        return 0.0
    smoothie = SmoothingFunction().method1
    return round(sentence_bleu([ref_tokens], cand_tokens, smoothing_function=smoothie), 3)

def generate_scores(item):
    scores = {field: round(random.uniform(0, 1), 3) for field in FIELDS[:-1]}
    # Use context as reference for BLEU if available, else 0
    reference = item.get("context", "")
    candidate = item.get("response", "")
    scores["BLEU"] = compute_bleu(reference, candidate)
    return scores

def main():
    with open(INPUT_FILE) as fin, open(OUTPUT_FILE, "w") as fout:
        for line in tqdm(fin):
            item = json.loads(line)
            item["llama2_scores"] = generate_scores(item)
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Synthetic scored file with BLEU written to {OUTPUT_FILE}")

if __name__ == "__main__":
    main() 