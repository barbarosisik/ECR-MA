import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm import tqdm
import argparse
import time
import re

MODEL_NAME = "/data1/s3905993/ECRHMAS/src/models/llama2_chat"  # Local path to Llama2-Chat model

def create_llama2_prompt(context, response):
    context_str = " ".join(context) if isinstance(context, list) else str(context)
    prompt = (
        "You are a strict and critical evaluator of conversational recommender systems. "
        "Your job is to rate the given system response using scores between 0.0 (worst) and 1.0 (best). "
        "Do NOT use any other format or scoring scale.\n\n"
        f"Given Context:\n{context_str}\n\n"
        f"System Response:\n{response}\n\n"
        "You MUST respond with EXACTLY this format:\n"
        "{\n"
        '"empathy_score": [score],\n'
        '"informativeness_score": [score],\n'
        '"recommendation_score": [score],\n'
        '"engagement_score": [score]\n'
        "}\n\n"
        "Then, on a new line, provide a single short sentence summarizing the response in overall.\n"
        "NO detailed explanations, no markdown, no code block, no quotes, no greetings, no closing statements."
    )
    return prompt

def parse_scores(result):
    import re
    
    # First, try to find complete JSON objects with braces
    json_matches = re.finditer(r'\{.*?\}', result, re.DOTALL)
    for match in json_matches:
        try:
            scores = json.loads(match.group())
            required_keys = ["empathy_score", "informativeness_score", "recommendation_score", "engagement_score"]
            if all(k in scores for k in required_keys):
                return {k: float(scores[k]) for k in required_keys}
        except Exception as e:
            continue
    
    # If no complete JSON found, try to extract key-value pairs and construct JSON
    try:
        # Look for the pattern: "key": value,
        pattern = r'"([^"]+)_score":\s*([0-9]*\.?[0-9]+)'
        matches = re.findall(pattern, result)
        
        if len(matches) >= 4:  # We need at least 4 scores
            scores = {}
            for key, value in matches:
                if key in ["empathy", "informativeness", "recommendation", "engagement"]:
                    scores[f"{key}_score"] = float(value)
            
            # Check if we have all required keys
            required_keys = ["empathy_score", "informativeness_score", "recommendation_score", "engagement_score"]
            if all(k in scores for k in required_keys):
                return scores
    except Exception as e:
        pass
    
    # Last resort: try to find any numbers that might be scores
    try:
        # Look for 4 decimal numbers that could be scores
        numbers = re.findall(r'([0-9]*\.?[0-9]+)', result)
        if len(numbers) >= 4:
            # Take the first 4 numbers as scores (in order: empathy, informativeness, recommendation, engagement)
            scores = {
                "empathy_score": float(numbers[0]),
                "informativeness_score": float(numbers[1]),
                "recommendation_score": float(numbers[2]),
                "engagement_score": float(numbers[3])
            }
            return scores
    except Exception as e:
        pass
    
    return None

def parse_explanation(result):
    """Extract the explanation sentence from the model output."""
    import re
    
    # Remove the JSON part and get everything after it
    # First, try to find where JSON ends
    json_end = None
    
    # Look for closing brace
    brace_match = re.search(r'\}', result)
    if brace_match:
        json_end = brace_match.end()
    
    # If no brace, look for the last score pattern
    if json_end is None:
        score_pattern = r'"engagement_score":\s*[0-9]*\.?[0-9]+'
        score_match = re.search(score_pattern, result)
        if score_match:
            json_end = score_match.end()
    
    if json_end:
        # Get everything after the JSON
        explanation_part = result[json_end:].strip()
        
        # Clean up the explanation
        explanation_part = re.sub(r'^\s*[,}\s]*', '', explanation_part)  # Remove leading commas, braces, whitespace
        explanation_part = re.sub(r'\s+', ' ', explanation_part)  # Normalize whitespace
        
        # Look for the first sentence that seems like an explanation
        sentences = explanation_part.split('.')
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence) > 10:  # Must be substantial
                # Skip if it's just numbers or technical text
                if not re.match(r'^[0-9\s\.]+$', sentence) and not sentence.lower().startswith('overall'):
                    return sentence
        
        # If no good sentence found, return the cleaned explanation part
        if explanation_part and len(explanation_part) > 10:
            return explanation_part
    
    return None

def get_llama2_scores(prompt_string, model, tokenizer):
    chat_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt_string}],
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = tokenizer(chat_prompt, return_tensors="pt").to(model.device)
    output = model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.5,
        top_p=0.9,
        eos_token_id=tokenizer.eos_token_id
    )
    result = tokenizer.decode(output[0], skip_special_tokens=True)
    assistant_reply = result[len(chat_prompt):].strip()
    scores = parse_scores(assistant_reply)
    explanation = parse_explanation(assistant_reply)
    return scores, explanation, assistant_reply 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='llama2_prompts.jsonl', help='Input file')
    parser.add_argument('--output', type=str, default='llama2_scored.jsonl', help='Output file')
    parser.add_argument('--max', type=int, default=None, help='Max number of prompts to process (for testing)')
    args = parser.parse_args()

    print("Loading model and tokenizer...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if device.type == 'cuda' else torch.float32,
        device_map=None if device.type == 'cpu' else 'auto'
    )
    print("Model loaded.")

    start_time = time.time()
    all_scores = []
    successful_parses = 0
    total_processed = 0
    
    with open(args.input) as fin, open(args.output, "w") as fout:
        for i, line in enumerate(fin):
            if args.max is not None and i >= args.max:
                break
                
            total_processed += 1
            item = json.loads(line)
            context = item.get("context", "")
            response = item.get("response", item.get("resp", ""))
            prompt = create_llama2_prompt(context, response)
            scores, explanation, full_output = get_llama2_scores(prompt, model, tokenizer)

            if scores:
                successful_parses += 1
                overall = sum(scores.values()) / 4.0
                scores["overall_score"] = round(overall, 4)
                item["llama2_scores"] = scores
                
                # Add explanation if available
                if explanation:
                    item["llama2_explanation"] = explanation
                else:
                    item["llama2_explanation"] = "No explanation available"
                
                fout.write(json.dumps(item, ensure_ascii=False) + "\n")
                all_scores.append(scores)
            else:
                print(f"\n[PARSE FAIL] Raw model output for sample {i+1}:\n{full_output}\n")
                # Still save the item with error info for debugging
                item["llama2_scores"] = {"error": "parsing failed"}
                item["llama2_explanation"] = "Parsing failed"
                fout.write(json.dumps(item, ensure_ascii=False) + "\n")

            if i < 5:
                print(f"\n=== Sample {i+1} ===")
                if scores:
                    print(json.dumps(scores, indent=2))
                    if explanation:
                        print(f"Explanation: {explanation}")
                else:
                    print(json.dumps({"error": "parsing failed"}, indent=2))

            # Progress update every 10 samples
            if (i + 1) % 10 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                parse_rate = (successful_parses / total_processed) * 100 if total_processed > 0 else 0
                print(f"[PROGRESS] {i+1} processed | Parse success: {parse_rate:.1f}% | Rate: {rate:.2f} samples/sec")

    print(f"\n===== FINAL STATISTICS =====")
    print(f"Total processed: {total_processed}")
    print(f"Successful parses: {successful_parses}")
    print(f"Parse success rate: {(successful_parses/total_processed)*100:.1f}%")
    
    print(f"\n===== SCORES FOR FIRST 5 RESPONSES =====")
    for idx, s in enumerate(all_scores[:5]):
        print(f"Sample {idx+1}: {s}")
    print("\nScoring complete.")

if __name__ == "__main__":
    main()