#!/bin/bash
#SBATCH --job-name=evaluate_ecr_comparison
#SBATCH --output=../slurm_outputs/evaluate_ecr_%j.out
#SBATCH --error=../slurm_outputs/evaluate_ecr_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-short

# Load modules
module load cuda/11.8
module load anaconda3

# Activate conda environment
source /data1/s3905993/conda_envs/ecrhmas_fixed/bin/activate

# Set environment variables for local cache
export HF_HOME="/data1/s3905993/cache/huggingface"
export TRANSFORMERS_CACHE="/data1/s3905993/cache/huggingface"

# Set working directory
cd /data1/s3905993/ECR-main

# Create output directories
mkdir -p data/evaluation_results
mkdir -p data/evaluation_logs

# Model configuration - can be changed for different experiments
MODEL_TYPE="dialogpt"  # Options: "dialogpt" or "llama2"

if [ "$MODEL_TYPE" = "dialogpt" ]; then
    echo "Evaluating DialoGPT model (ECR baseline comparison)"
    MODEL_NAME="microsoft/DialoGPT-small"
    TOKENIZER_NAME="microsoft/DialoGPT-small"
    MODEL_PATH="data/saved/rl_models/dialogpt_rl"
    OUTPUT_DIR="data/evaluation_results/dialogpt_eval"
elif [ "$MODEL_TYPE" = "llama2" ]; then
    echo "Evaluating Llama2-chat model (ECR comparison)"
    MODEL_NAME="/data1/s3905993/ECRHMAS/src/models/llama2_chat"
    TOKENIZER_NAME="/data1/s3905993/ECRHMAS/src/models/llama2_chat"
    MODEL_PATH="data/saved/rl_models/llama2_rl"
    OUTPUT_DIR="data/evaluation_results/llama2_eval"
else
    echo "Invalid model type. Please choose 'dialogpt' or 'llama2'"
    exit 1
fi

# Run evaluation with ECR-compatible metrics
python src_emo/evaluate_rl.py \
    --dataset redial \
    --model_path $MODEL_PATH \
    --model_name $MODEL_NAME \
    --tokenizer $TOKENIZER_NAME \
    --text_tokenizer roberta-base \
    --output_dir $OUTPUT_DIR \
    --test_data data/processed/test_data.json \
    --batch_size 8 \
    --max_gen_len 150 \
    --metrics bleu,distinct,hit@k,mrr@k,ndcg@k \
    --k_values 1,5,10 \
    --seed 42

# Generate ECR-style evaluation report
echo "Generating ECR comparison report..."
python -c "
import json
import os
from datetime import datetime

# Load evaluation results
eval_file = '$OUTPUT_DIR/evaluation_results.json'
if os.path.exists(eval_file):
    with open(eval_file, 'r') as f:
        results = json.load(f)
    
    # Create ECR-style report
    report = {
        'model_type': '$MODEL_TYPE',
        'evaluation_date': datetime.now().isoformat(),
        'metrics': results,
        'ecr_comparison': {
            'recommendation_metrics': {
                'AUC': results.get('auc', 'N/A'),
                'RT@1': results.get('recall_true@1', 'N/A'),
                'RT@10': results.get('recall_true@10', 'N/A'),
                'RT@50': results.get('recall_true@50', 'N/A')
            },
            'generation_metrics': {
                'BLEU': results.get('bleu', 'N/A'),
                'Distinct': results.get('distinct', 'N/A'),
                'HIT@1': results.get('hit@1', 'N/A'),
                'HIT@5': results.get('hit@5', 'N/A'),
                'HIT@10': results.get('hit@10', 'N/A'),
                'MRR@1': results.get('mrr@1', 'N/A'),
                'MRR@5': results.get('mrr@5', 'N/A'),
                'MRR@10': results.get('mrr@10', 'N/A'),
                'NDCG@1': results.get('ndcg@1', 'N/A'),
                'NDCG@5': results.get('ndcg@5', 'N/A'),
                'NDCG@10': results.get('ndcg@10', 'N/A')
            }
        }
    }
    
    # Save ECR comparison report
    report_file = '$OUTPUT_DIR/ecr_comparison_report.json'
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f'ECR comparison report saved to: {report_file}')
    print(f'Model: {MODEL_TYPE}')
    print(f'Recommendation AUC: {results.get(\"auc\", \"N/A\")}')
    print(f'Generation BLEU: {results.get(\"bleu\", \"N/A\")}')
else:
    print('Evaluation results not found. Please run evaluation first.')
"

echo "Evaluation completed successfully for $MODEL_TYPE!" 