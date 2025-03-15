#!/bin/bash

# Script to run GUE evaluation on ModernDNABERT
# Usage: ./run_gue_evaluation.sh /path/to/model /path/to/tokenizer /path/to/gue_data

# Set default values
MODEL_PATH=$1
TOKENIZER_PATH=$2
GUE_PATH=$3
OUTPUT_DIR="./results"
USE_ALIBI=true
BATCH_SIZE=8
NUM_GPUS=1

# Check if required arguments are provided
if [ -z "$MODEL_PATH" ] || [ -z "$TOKENIZER_PATH" ] || [ -z "$GUE_PATH" ]; then
    echo "Usage: $0 MODEL_PATH TOKENIZER_PATH GUE_PATH [OUTPUT_DIR] [NUM_GPUS]"
    echo "Example: $0 ./output/model ./output/tokenizer ./GUE ./results 4"
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

# Function to run evaluation on a task
run_task() {
    TASK_NAME=$1
    MAX_LENGTH=$2
    LEARNING_RATE=$3
    EPOCHS=$4

    echo "Evaluating on $TASK_NAME..."

    # If multiple GPUs are available, use DistributedDataParallel
    if [ $NUM_GPUS -gt 1 ]; then
        CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS evaluate_gue.py \
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

echo "Starting GUE evaluation with ModernDNABERT..."
echo "Model path: $MODEL_PATH"
echo "Tokenizer path: $TOKENIZER_PATH"
echo "GUE data path: $GUE_PATH"
echo "Output directory: $OUTPUT_DIR"
echo "Number of GPUs: $NUM_GPUS"

# EMP tasks
for task in H3 H3K14ac H3K36me3 H3K4me1 H3K4me2 H3K4me3 H3K79me3 H3K9ac H4 H4ac; do
    run_task "emp_$task" 128 3e-5 3
done

# Promoter tasks
run_task "prom_core_all" 20 3e-5 4
run_task "prom_core_notata" 20 3e-5 4
run_task "prom_core_tata" 20 3e-5 10
run_task "prom_300_all" 70 3e-5 4
run_task "prom_300_notata" 70 3e-5 4
run_task "prom_300_tata" 70 3e-5 10

# Splice site task
run_task "splice_reconstructed" 80 3e-5 5

# Virus task
run_task "virus_covid" 256 3e-5 8

# Mouse tasks
for i in {0..4}; do
    run_task "mouse_$i" 30 3e-5 5
done

# Transcription factor tasks
for i in {0..4}; do
    run_task "tf_$i" 30 3e-5 3
done

echo "GUE evaluation completed. Results saved to $OUTPUT_DIR"

# Generate summary report
python -c "
import json
import os

output_dir = '$OUTPUT_DIR'
all_results = {}

for task_dir in os.listdir(output_dir):
    result_path = os.path.join(output_dir, task_dir, 'results', 'test_results.json')
    if os.path.exists(result_path):
        with open(result_path, 'r') as f:
            results = json.load(f)
            all_results[task_dir] = results

# Calculate average metrics
metrics = ['eval_accuracy', 'eval_f1', 'eval_matthews_correlation', 'eval_precision', 'eval_recall']
avg_metrics = {metric: 0 for metric in metrics}
count = 0

for task, results in all_results.items():
    count += 1
    for metric in metrics:
        if metric in results:
            avg_metrics[metric] += results[metric]

for metric in avg_metrics:
    if count > 0:
        avg_metrics[metric] /= count

# Print summary
print('\nGUE Benchmark Summary:')
print(f'Total tasks evaluated: {count}/28')
print('\nAverage metrics:')
for metric in metrics:
    print(f'{metric[5:]}: {avg_metrics[metric]:.4f}')

# Save summary
with open(os.path.join(output_dir, 'summary.json'), 'w') as f:
    json.dump({
        'total_tasks': count,
        'average_metrics': avg_metrics,
        'task_results': all_results
    }, f, indent=4)

print(f'\nSummary saved to {os.path.join(output_dir, 'summary.json')}')
"

echo "GUE evaluation summary generated."