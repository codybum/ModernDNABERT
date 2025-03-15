#!/bin/bash

# Optimized script to run GUE evaluation on a genomic BERT model with ALiBi attention
# Usage: ./run_gue_evaluation.sh /path/to/model /path/to/tokenizer /path/to/gue_data [OUTPUT_DIR] [NUM_GPUS]
# Example: ./run_gue_evaluation.sh ./genomic_bert_model/final ./genomic_bert_model/tokenizer ./GUE ./results 4

# Set default values
MODEL_PATH=$1
TOKENIZER_PATH=$2
GUE_PATH=$3
OUTPUT_DIR="./gue_results"
USE_ALIBI=true
NUM_GPUS=8  # Using 8 GPUs as reported

# Check if required arguments are provided
if [ -z "$MODEL_PATH" ] || [ -z "$TOKENIZER_PATH" ] || [ -z "$GUE_PATH" ]; then
    echo "Usage: $0 MODEL_PATH TOKENIZER_PATH GUE_PATH [OUTPUT_DIR] [NUM_GPUS]"
    echo "Example: $0 ./genomic_bert_model/final ./genomic_bert_model/tokenizer ./GUE ./results 4"
    exit 1
fi

# Use 4th argument as output dir if provided
if [ ! -z "$4" ]; then
    OUTPUT_DIR=$4
fi

# Use 5th argument as number of GPUs if provided
if [ ! -z "$5" ]; then
    NUM_GPUS=$5
fi

# Create output directory
mkdir -p $OUTPUT_DIR

# Function to run evaluation on a task with optimized parameters
run_task() {
    TASK_NAME=$1
    MAX_LENGTH=$2
    BATCH_SIZE=$3
    LEARNING_RATE=$4
    EPOCHS=$5

    echo "==================================================================="
    echo "Evaluating on $TASK_NAME with max_length=$MAX_LENGTH, batch_size=$BATCH_SIZE"
    echo "==================================================================="

    # If multiple GPUs are available, use DistributedDataParallel
    if [ $NUM_GPUS -gt 1 ]; then
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python evaluate_gue.py \
            --model_path $MODEL_PATH \
            --tokenizer_path $TOKENIZER_PATH \
            --gue_path $GUE_PATH \
            --output_dir $OUTPUT_DIR \
            --tasks $TASK_NAME \
            --max_length $MAX_LENGTH \
            --batch_size $BATCH_SIZE \
            --learning_rate $LEARNING_RATE \
            --epochs $EPOCHS \
            --use_alibi
    else
        python evaluate_gue.py \
            --model_path $MODEL_PATH \
            --tokenizer_path $TOKENIZER_PATH \
            --gue_path $GUE_PATH \
            --output_dir $OUTPUT_DIR \
            --tasks $TASK_NAME \
            --max_length $MAX_LENGTH \
            --batch_size $BATCH_SIZE \
            --learning_rate $LEARNING_RATE \
            --epochs $EPOCHS \
            --use_alibi
    fi
}

echo "Starting optimized GUE evaluation for ALiBi model with 8 GPUs (each with 144GB VRAM)..."
echo "Model path: $MODEL_PATH"
echo "Tokenizer path: $TOKENIZER_PATH"
echo "GUE data path: $GUE_PATH"
echo "Output directory: $OUTPUT_DIR"
echo "Number of GPUs: $NUM_GPUS"
echo "Using ALiBi attention: $USE_ALIBI"
echo "Model was trained with pre_training_length=6144, max_inference_length=24576"
echo "Current configuration uses massive batch sizes and sequence lengths to utilize GPU resources"

# Calculate batch size based on having 8 GPUs with 144GB VRAM each
# With such massive GPU memory and low utilization reported, we need to dramatically increase batch sizes
BASE_BATCH_SIZE=128  # Per GPU for medium-length sequences - using much larger batches

# EMP tasks - originally 128, now using 12288 (double pre-training length) to test extrapolation
# With 8x144GB GPUs, we can push sequence lengths far beyond pre-training
for task in H3 H3K14ac H3K36me3 H3K4me1 H3K4me2 H3K4me3 H3K79me3 H3K9ac H4 H4ac; do
    run_task "emp_$task" 12288 $((BASE_BATCH_SIZE/2)) 2e-5 3  # 64 per GPU for 12k sequences with 8 GPUs
done

# Promoter core tasks - originally 20, we'll use 2048 for much more context
# These short classification tasks can still use very large batch sizes
run_task "prom_core_all" 2048 $BASE_BATCH_SIZE 3e-5 4  # Full 128 per GPU
run_task "prom_core_notata" 2048 $BASE_BATCH_SIZE 3e-5 4
run_task "prom_core_tata" 2048 $BASE_BATCH_SIZE 3e-5 6  # More epochs for tata tasks

# Promoter 300 tasks - originally 70, we'll use 8192 to provide massive context
run_task "prom_300_all" 8192 $((BASE_BATCH_SIZE/2)) 3e-5 4  # 64 per GPU
run_task "prom_300_notata" 8192 $((BASE_BATCH_SIZE/2)) 3e-5 4
run_task "prom_300_tata" 8192 $((BASE_BATCH_SIZE/2)) 3e-5 6  # More epochs for tata tasks

# Splice site task - originally 80, we'll use 12288 (double pre-training length)
# With 8x H200 GPUs, we can test extreme extrapolation for splice contexts
run_task "splice_reconstructed" 12288 $((BASE_BATCH_SIZE/2)) 2e-5 5  # 64 per GPU

# Virus task - originally 256, we'll use 24576 (max_inference_length)
# With 8x H200 GPUs, we can test the absolute maximum extrapolation capability
# Virus genomes are long and will benefit enormously from this length
run_task "virus_covid" 24576 $((BASE_BATCH_SIZE/4)) 2e-5 6  # 32 per GPU even for maximum length sequences

# Mouse tasks - originally 30, we'll use 8192 to test extensive genomic context
# With 8x H200 GPUs, we can provide massive flanking regions
for i in {0..4}; do
    run_task "mouse_$i" 8192 $((BASE_BATCH_SIZE/2)) 3e-5 5  # 64 per GPU
done

# Transcription factor tasks - originally 30, we'll use 8192
# TF binding predictions greatly benefit from extensive sequence context
for i in {0..4}; do
    run_task "tf_$i" 8192 $((BASE_BATCH_SIZE/2)) 3e-5 4  # 64 per GPU
done

echo "GUE evaluation completed. Results saved to $OUTPUT_DIR"

# Generate summary report
python -c "
import json
import os
import numpy as np

output_dir = '$OUTPUT_DIR'
all_results = {}
task_categories = {
    'emp_': 'Epigenetic Marker Prediction',
    'prom_': 'Promoter Identification',
    'splice_': 'Splice Site Prediction',
    'virus_': 'Virus Classification',
    'mouse_': 'Mouse-Specific Tasks',
    'tf_': 'Transcription Factor Binding'
}

print('\n' + '='*80)
print('GUE BENCHMARK EVALUATION SUMMARY'.center(80))
print('='*80)

# Process each task directory
for task_dir in os.listdir(output_dir):
    result_path = os.path.join(output_dir, task_dir, 'results', 'test_results.json')
    if os.path.exists(result_path):
        with open(result_path, 'r') as f:
            results = json.load(f)
            all_results[task_dir] = results

# Group tasks by category
category_results = {}
for task, results in all_results.items():
    # Find the category this task belongs to
    category = None
    for prefix, category_name in task_categories.items():
        if task.startswith(prefix):
            category = category_name
            break

    if category is None:
        category = 'Other'

    if category not in category_results:
        category_results[category] = []

    category_results[category].append((task, results))

# Calculate average metrics
metrics = ['eval_accuracy', 'eval_f1', 'eval_matthews_correlation', 'eval_precision', 'eval_recall']
overall_avg_metrics = {metric: [] for metric in metrics}
tasks_evaluated = 0

# Print results by category
for category, task_results in category_results.items():
    print(f'\n{category} Tasks:')
    print('-' * 80)
    print(f'Task'.ljust(25) + ' | ' + 'Accuracy'.center(10) + ' | ' + 'F1'.center(10) + ' | ' + 'MCC'.center(10))
    print('-' * 80)

    category_metrics = {metric: [] for metric in metrics}

    for task, results in task_results:
        tasks_evaluated += 1
        acc = results.get('eval_accuracy', 'N/A')
        f1 = results.get('eval_f1', 'N/A')
        mcc = results.get('eval_matthews_correlation', 'N/A')

        print(f'{task.ljust(25)} | {str(round(acc*100, 2) if acc != \"N/A\" else \"N/A\").center(10)} | {str(round(f1*100, 2) if f1 != \"N/A\" else \"N/A\").center(10)} | {str(round(mcc*100, 2) if mcc != \"N/A\" else \"N/A\").center(10)}')

        # Collect metrics for averaging
        for metric in metrics:
            if metric in results:
                category_metrics[metric].append(results[metric])
                overall_avg_metrics[metric].append(results[metric])

    # Print category averages
    print('-' * 80)
    category_avg = {metric: np.mean(values) if values else float('nan') for metric, values in category_metrics.items()}
    print(f'CATEGORY AVERAGE:'.ljust(25) + ' | ' +
          f'{str(round(category_avg[\"eval_accuracy\"]*100, 2) if not np.isnan(category_avg[\"eval_accuracy\"]) else \"N/A\").center(10)} | ' +
          f'{str(round(category_avg[\"eval_f1\"]*100, 2) if not np.isnan(category_avg[\"eval_f1\"]) else \"N/A\").center(10)} | ' +
          f'{str(round(category_avg[\"eval_matthews_correlation\"]*100, 2) if not np.isnan(category_avg[\"eval_matthews_correlation\"]) else \"N/A\").center(10)}')

# Calculate and print overall averages
print('\n' + '='*80)
print('OVERALL BENCHMARK RESULTS'.center(80))
print('='*80)
print(f'Total tasks evaluated: {tasks_evaluated}/28')
print('\nAverage performance across all tasks:')
for metric in metrics:
    if overall_avg_metrics[metric]:
        avg_value = np.mean(overall_avg_metrics[metric])
        if metric == 'eval_accuracy':
            print(f'Accuracy:  {avg_value*100:.2f}%')
        elif metric == 'eval_f1':
            print(f'F1 Score:  {avg_value*100:.2f}%')
        elif metric == 'eval_matthews_correlation':
            print(f'Matthews:  {avg_value*100:.2f}%')
        elif metric == 'eval_precision':
            print(f'Precision: {avg_value*100:.2f}%')
        elif metric == 'eval_recall':
            print(f'Recall:    {avg_value*100:.2f}%')

# Comparison context (typical SOTA ranges)
print('\nContext: SOTA performance ranges for genomic tasks:')
print('- Epigenetic marker prediction: 70-85% accuracy')
print('- Promoter identification: 85-95% accuracy')
print('- Splice site prediction: 90-97% accuracy')
print('- Transcription factor binding: 75-90% F1 score')

# Save summary with more detailed metrics
with open(os.path.join(output_dir, 'detailed_summary.json'), 'w') as f:
    json.dump({
        'total_tasks': tasks_evaluated,
        'overall_metrics': {metric: np.mean(values).item() if values else None for metric, values in overall_avg_metrics.items()},
        'category_metrics': {cat: {metric: np.mean([res[metric] for _, res in task_results if metric in res]).item() if any(metric in res for _, res in task_results) else None for metric in metrics} for cat, task_results in category_results.items()},
        'task_results': all_results
    }, f, indent=4)

print(f'\nDetailed summary saved to {os.path.join(output_dir, \"detailed_summary.json\")}')
"

echo "GUE evaluation summary generated."