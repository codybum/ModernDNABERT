#!/usr/bin/env python
"""
GUE evaluation script for ModernDNABERT.

This script fine-tunes and evaluates ModernDNABERT on the GUE benchmark,
using the ALiBi attention mechanism for improved sequence handling.
"""
import os
import argparse
import logging
import json
import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, precision_score, recall_score
from typing import Optional

from torch.utils.data import Dataset
from transformers import (
    Trainer, TrainingArguments, PretrainedConfig,
    set_seed
)

# Import your custom modules here
from modeling.alibi_attention import create_genomic_bert_config, create_genomic_bert_model
from tokenization.genomic_tokenizer import GenomicTokenizer

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger(__name__)


class GUEDataset(Dataset):
    """Dataset for GUE tasks."""

    def __init__(self, data_path, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Load data from CSV
        import csv
        with open(data_path, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            data = list(reader)

        self.sequences = [row[0] for row in data]
        self.labels = [int(row[1]) for row in data]
        self.num_labels = len(set(self.labels))

        logger.info(f"Loaded {len(self.sequences)} examples from {data_path}")
        logger.info(f"Number of labels: {self.num_labels}")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]

        # Only use the characters A, T, G, C
        sequence = ''.join(c for c in sequence.upper() if c in 'ATGC')

        # Tokenize sequence
        encoding = self.tokenizer(
            sequence,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )

        # Remove batch dimension
        encoding = {k: v.squeeze(0) for k, v in encoding.items()}

        # Add label
        encoding["labels"] = torch.tensor(label)

        return encoding


def compute_metrics(eval_pred):
    """Compute metrics for evaluation."""
    predictions, labels = eval_pred

    # Get predicted class from logits
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    predictions = np.argmax(predictions, axis=1)

    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average="macro", zero_division=0),
        "matthews_correlation": matthews_corrcoef(labels, predictions),
        "precision": precision_score(labels, predictions, average="macro", zero_division=0),
        "recall": recall_score(labels, predictions, average="macro", zero_division=0)
    }


class ModernDNABERTForSequenceClassification(torch.nn.Module):
    """
    Classification model that adapts a BertForMaskedLM to do sequence classification.
    """

    def __init__(self, bert_model, num_labels):
        super().__init__()
        self.bert = bert_model
        self.num_labels = num_labels

        # Add classification head
        self.dropout = torch.nn.Dropout(self.bert.config.hidden_dropout_prob)
        self.classifier = torch.nn.Linear(self.bert.config.hidden_size, num_labels)

        # Initialize weights
        self.classifier.weight.data.normal_(mean=0.0, std=0.02)
        if self.classifier.bias is not None:
            self.classifier.bias.data.zero_()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                labels=None, return_dict=None, **kwargs):
        """
        Forward pass that adapts BERT MLM model to sequence classification.
        Handles extra kwargs that might be passed by Trainer.
        """
        # Forward pass through BERT, but handle the case that bert is BertForMaskedLM
        # We need to get the hidden states without the MLM head
        bert_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
            output_hidden_states=True,  # Request hidden states explicitly
        )

        # For BertForMaskedLM, we need to access the hidden states differently
        if hasattr(bert_outputs, 'hidden_states') and bert_outputs.hidden_states is not None:
            # If hidden_states is available, use the last layer
            sequence_output = bert_outputs.hidden_states[-1]
        elif hasattr(bert_outputs, 'last_hidden_state'):
            # If last_hidden_state is available (typical for BertModel)
            sequence_output = bert_outputs.last_hidden_state
        else:
            # If we can't get the hidden states, try to access the internal BERT model
            try:
                # BertForMaskedLM typically has a .bert attribute that's a BertModel
                internal_bert_outputs = self.bert.bert(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    return_dict=True,
                )
                sequence_output = internal_bert_outputs.last_hidden_state
            except AttributeError:
                raise ValueError(
                    "Cannot extract sequence output from model. Make sure it provides either hidden_states or last_hidden_state.")

        # Use the [CLS] token for classification
        pooled_output = sequence_output[:, 0]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        # Return in format expected by HF Trainer
        from transformers.modeling_outputs import SequenceClassifierOutput
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=bert_outputs.hidden_states if hasattr(bert_outputs, 'hidden_states') else None,
            attentions=bert_outputs.attentions if hasattr(bert_outputs, 'attentions') else None,
        )


class GenomicTrainer(Trainer):
    """
    Custom Trainer class that properly handles shared tensors when saving with SafeTensors.
    This fixes the shared memory error when saving BERT models.
    """

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        """
        Override the _save method to handle shared weights in BERT models.
        This prevents the SafeTensors error with shared memory tensors.
        """
        # Warn if not saving to output_dir
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Get the state dict if not provided
        if state_dict is None:
            state_dict = self.model.state_dict()

        # Handle shared weights by creating copies
        cleaned_state_dict = {}
        for key, value in state_dict.items():
            # Create a copy of each tensor so they don't share memory
            cleaned_state_dict[key] = value.clone()

        # Use PyTorch's native save instead of SafeTensors
        # This avoids issues with shared memory tensors
        # logger = self.get_logger()
        # logger.info(f"Saving model to {output_dir}")
        torch.save(cleaned_state_dict, os.path.join(output_dir, "pytorch_model.bin"))

        # Save the config
        if hasattr(self.model, "config"):
            self.model.config.save_pretrained(output_dir)

        # Return the output directory
        return output_dir

def load_model_for_classification(model_path, tokenizer, num_labels):
    """
    Load a model with correct handling for ALiBi models and distributed training.
    This function is designed to be compatible with both single-GPU and multi-GPU setups.
    """
    try:
        # Import SafeTensors if available
        from safetensors.torch import load_file
        has_safetensors = True
    except ImportError:
        logger.warning("SafeTensors not installed. Using PyTorch loading instead.")
        has_safetensors = False

    # Step 1: Load the original config
    config_path = os.path.join(model_path, "config.json")
    if os.path.exists(config_path):
        logger.info(f"Loading configuration from {config_path}")
        config = PretrainedConfig.from_json_file(config_path)

        # Store the original position embeddings size
        original_max_position = getattr(config, 'max_position_embeddings', 512)
        logger.info(f"Original max_position_embeddings: {original_max_position}")

        # Set max_position_embeddings to 512 for classification tasks
        # But don't change it yet - we'll handle this in the state dict
        config.max_position_embeddings = original_max_position
    else:
        # If no config, create one with default position embeddings size
        logger.info(f"No config.json found, creating default config")
        config = create_genomic_bert_config(
            vocab_size=tokenizer.vocab_size,
            max_position_embeddings=512,
            use_alibi=True,
            attention_type="alibi"
        )
        original_max_position = 512

    # Step 2: Check if the model uses ALiBi attention
    uses_alibi = getattr(config, 'use_alibi', False) or getattr(config, 'attention_type', '') == 'alibi'
    if uses_alibi:
        logger.info("Model uses ALiBi attention")
        # For ALiBi models, we'll keep the position embeddings but they won't be used

    # Step 3: Load state dict first to inspect/modify it before creating model
    state_dict = None
    safetensors_path = os.path.join(model_path, "model.safetensors")
    pytorch_path = os.path.join(model_path, "pytorch_model.bin")

    if os.path.exists(safetensors_path) and has_safetensors:
        logger.info(f"Loading model weights from SafeTensors: {safetensors_path}")
        try:
            state_dict = load_file(safetensors_path)
        except Exception as e:
            logger.error(f"Error loading SafeTensors file: {e}")
            if os.path.exists(pytorch_path):
                logger.info(f"Falling back to PyTorch weights: {pytorch_path}")
                state_dict = torch.load(pytorch_path, map_location="cpu")
    elif os.path.exists(pytorch_path):
        logger.info(f"Loading model weights from PyTorch: {pytorch_path}")
        state_dict = torch.load(pytorch_path, map_location="cpu")
    else:
        raise FileNotFoundError(f"No model weights found at {model_path}")

    # Step 4: Modify config for creating the model
    # For classification, we use a smaller position embedding size
    if original_max_position != 512:
        logger.info(f"Setting max_position_embeddings to " + str(original_max_position) + " for classification model")
        config.max_position_embeddings = original_max_position

    # Step 5: Create model with modified config
    logger.info("Creating base BERT model")
    bert_model = create_genomic_bert_model(config)

    # Step 6: Handle the position embeddings in state_dict for ALiBi models
    if 'bert.embeddings.position_embeddings.weight' in state_dict:
        pos_embed_weight = state_dict['bert.embeddings.position_embeddings.weight']
        pretrained_size = pos_embed_weight.shape[0]

        if pretrained_size != 512 and uses_alibi:
            logger.info(f"ALiBi model: Handling position embeddings tensor of size {pretrained_size}")
            # Option 1: Remove position embeddings from state dict for ALiBi models
            del state_dict['bert.embeddings.position_embeddings.weight']
            logger.info("Removed position embeddings from state dict for ALiBi model")

    # Step 7: Load weights into model
    logger.info("Loading weights into model")
    # Use strict=False to allow missing position embeddings in ALiBi models
    missing_keys, unexpected_keys = bert_model.load_state_dict(state_dict, strict=False)

    # Log missing and unexpected keys
    if missing_keys:
        logger.info(f"Missing keys: {missing_keys}")
    if unexpected_keys:
        logger.info(f"Unexpected keys: {unexpected_keys}")

    # Step 8: Create sequence classification model
    logger.info(f"Creating sequence classification model with {num_labels} labels")
    model = ModernDNABERTForSequenceClassification(bert_model, num_labels)

    # Step 9: Add DDP-specific settings to model - with SAFER ATTRIBUTE ACCESS
    if uses_alibi:
        # Carefully check each level of the attribute hierarchy
        if hasattr(model, 'bert'):
            bert_module = model.bert
            # Next check if bert has embeddings
            if hasattr(bert_module, 'embeddings'):
                embeddings_module = bert_module.embeddings
                # Finally check for position_embeddings
                if hasattr(embeddings_module, 'position_embeddings'):
                    # Freeze position embeddings to prevent DDP sync issues
                    embeddings_module.position_embeddings.requires_grad = False
                    logger.info("Froze position embeddings for distributed training compatibility")

    return model

def evaluate_on_task(model_path, tokenizer_path, task_path, output_dir, task_name, max_length=512,
                     batch_size=16, learning_rate=3e-5, epochs=3, seed=42, use_alibi=True):
    """Evaluate the model on a single GUE task."""
    # Set random seed
    set_seed(seed)

    # Load tokenizer
    logger.info(f"Loading tokenizer from {tokenizer_path}")
    tokenizer = GenomicTokenizer.from_pretrained(tokenizer_path)

    # Load datasets
    train_dataset = GUEDataset(
        os.path.join(task_path, "train.csv"),
        tokenizer,
        max_length=max_length
    )

    val_dataset = GUEDataset(
        os.path.join(task_path, "dev.csv"),
        tokenizer,
        max_length=max_length
    )

    test_dataset = GUEDataset(
        os.path.join(task_path, "test.csv"),
        tokenizer,
        max_length=max_length
    )

    # Create model for classification
    logger.info(f"Loading model from {model_path} for classification with {train_dataset.num_labels} labels")
    model = load_model_for_classification(model_path, tokenizer, train_dataset.num_labels)

    # Set up training arguments
    task_output_dir = os.path.join(output_dir, task_name)
    training_args = TrainingArguments(
        output_dir=task_output_dir,
        eval_strategy="steps",  # Use eval_strategy instead of deprecated evaluation_strategy
        eval_steps=100,
        save_steps=100,
        save_total_limit=2,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="eval_matthews_correlation",
        fp16=torch.cuda.is_available(),
        logging_dir=os.path.join(task_output_dir, "logs"),
        logging_steps=50,
        report_to="none",
        # Add these options to avoid SafeTensors error
        save_safetensors=False,  # Disable SafeTensors saving
    )

    # Set up trainer with custom trainer class that handles shared weights
    trainer = GenomicTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    # Train the model
    logger.info(f"Starting training for {task_name}")
    trainer.train()

    # Evaluate on test set
    logger.info(f"Evaluating {task_name} on test set")
    results = trainer.evaluate(test_dataset)

    # Save results
    os.makedirs(os.path.join(task_output_dir, "results"), exist_ok=True)
    with open(os.path.join(task_output_dir, "results", "test_results.json"), "w") as f:
        json.dump(results, f, indent=4)

    logger.info(f"Results for {task_name}: {results}")

    return results

def main():
    parser = argparse.ArgumentParser(description="Evaluate ModernDNABERT on GUE benchmark")

    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to pre-trained ModernDNABERT model")
    parser.add_argument("--tokenizer_path", type=str, required=True,
                        help="Path to tokenizer")
    parser.add_argument("--gue_path", type=str, required=True,
                        help="Path to GUE benchmark directory")
    parser.add_argument("--output_dir", type=str, default="./gue_results",
                        help="Directory to save results")

    # Task selection
    parser.add_argument("--tasks", type=str, nargs="+",
                        default=["all"],
                        help="Tasks to evaluate on, 'all' for all tasks")

    # Training parameters
    parser.add_argument("--max_length", type=int, default=512,
                        help="Maximum sequence length")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for training and evaluation")
    parser.add_argument("--learning_rate", type=float, default=3e-5,
                        help="Learning rate")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--use_alibi", action="store_true",
                        help="Use ALiBi attention mechanism")

    args = parser.parse_args()

    # Define all available tasks
    all_tasks = {
        # EMP tasks
        "emp_H3": {"path": os.path.join(args.gue_path, "GUE/EMP/H3"), "max_length": 128},
        "emp_H3K14ac": {"path": os.path.join(args.gue_path, "GUE/EMP/H3K14ac"), "max_length": 128},
        "emp_H3K36me3": {"path": os.path.join(args.gue_path, "GUE/EMP/H3K36me3"), "max_length": 128},
        "emp_H3K4me1": {"path": os.path.join(args.gue_path, "GUE/EMP/H3K4me1"), "max_length": 128},
        "emp_H3K4me2": {"path": os.path.join(args.gue_path, "GUE/EMP/H3K4me2"), "max_length": 128},
        "emp_H3K4me3": {"path": os.path.join(args.gue_path, "GUE/EMP/H3K4me3"), "max_length": 128},
        "emp_H3K79me3": {"path": os.path.join(args.gue_path, "GUE/EMP/H3K79me3"), "max_length": 128},
        "emp_H3K9ac": {"path": os.path.join(args.gue_path, "GUE/EMP/H3K9ac"), "max_length": 128},
        "emp_H4": {"path": os.path.join(args.gue_path, "GUE/EMP/H4"), "max_length": 128},
        "emp_H4ac": {"path": os.path.join(args.gue_path, "GUE/EMP/H4ac"), "max_length": 128},

        # Promoter tasks
        "prom_core_all": {"path": os.path.join(args.gue_path, "GUE/prom/prom_core_all"), "max_length": 20},
        "prom_core_notata": {"path": os.path.join(args.gue_path, "GUE/prom/prom_core_notata"), "max_length": 20},
        "prom_core_tata": {"path": os.path.join(args.gue_path, "GUE/prom/prom_core_tata"), "max_length": 20},
        "prom_300_all": {"path": os.path.join(args.gue_path, "GUE/prom/prom_300_all"), "max_length": 70},
        "prom_300_notata": {"path": os.path.join(args.gue_path, "GUE/prom/prom_300_notata"), "max_length": 70},
        "prom_300_tata": {"path": os.path.join(args.gue_path, "GUE/prom/prom_300_tata"), "max_length": 70},

        # Splice site task
        "splice_reconstructed": {"path": os.path.join(args.gue_path, "GUE/splice/reconstructed"), "max_length": 80},

        # Virus task
        "virus_covid": {"path": os.path.join(args.gue_path, "GUE/virus/covid"), "max_length": 256},

        # Mouse tasks
        "mouse_0": {"path": os.path.join(args.gue_path, "GUE/mouse/0"), "max_length": 30},
        "mouse_1": {"path": os.path.join(args.gue_path, "GUE/mouse/1"), "max_length": 30},
        "mouse_2": {"path": os.path.join(args.gue_path, "GUE/mouse/2"), "max_length": 30},
        "mouse_3": {"path": os.path.join(args.gue_path, "GUE/mouse/3"), "max_length": 30},
        "mouse_4": {"path": os.path.join(args.gue_path, "GUE/mouse/4"), "max_length": 30},

        # Transcription factor tasks
        "tf_0": {"path": os.path.join(args.gue_path, "GUE/tf/0"), "max_length": 30},
        "tf_1": {"path": os.path.join(args.gue_path, "GUE/tf/1"), "max_length": 30},
        "tf_2": {"path": os.path.join(args.gue_path, "GUE/tf/2"), "max_length": 30},
        "tf_3": {"path": os.path.join(args.gue_path, "GUE/tf/3"), "max_length": 30},
        "tf_4": {"path": os.path.join(args.gue_path, "GUE/tf/4"), "max_length": 30},
    }

    # Determine which tasks to run
    tasks_to_run = all_tasks.keys() if "all" in args.tasks else args.tasks

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Store all results
    all_results = {}

    # Run evaluation on selected tasks
    for task_name in tasks_to_run:
        if task_name not in all_tasks:
            logger.warning(f"Task {task_name} not found, skipping.")
            continue

        task_info = all_tasks[task_name]
        max_length = task_info.get("max_length", args.max_length)

        logger.info(f"Evaluating on task: {task_name}")
        try:
            results = evaluate_on_task(
                model_path=args.model_path,
                tokenizer_path=args.tokenizer_path,
                task_path=task_info["path"],
                output_dir=args.output_dir,
                task_name=task_name,
                max_length=max_length,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                epochs=args.epochs,
                seed=args.seed,
                use_alibi=args.use_alibi
            )

            all_results[task_name] = results
        except Exception as e:
            logger.error(f"Error evaluating task {task_name}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            logger.error("Continuing with next task")

    # Save overall results
    with open(os.path.join(args.output_dir, "all_results.json"), "w") as f:
        json.dump(all_results, f, indent=4)

    # Calculate and display average metrics
    avg_metrics = {
        "accuracy": 0,
        "f1": 0,
        "matthews_correlation": 0,
        "precision": 0,
        "recall": 0
    }

    for task_results in all_results.values():
        for metric in avg_metrics:
            metric_key = f"eval_{metric}"
            if metric_key in task_results:
                avg_metrics[metric] += task_results[metric_key]

    if all_results:
        for metric in avg_metrics:
            avg_metrics[metric] /= len(all_results)

        logger.info(f"Average metrics across all tasks:")
        for metric, value in avg_metrics.items():
            logger.info(f"  {metric}: {value:.4f}")

        with open(os.path.join(args.output_dir, "average_metrics.json"), "w") as f:
            json.dump(avg_metrics, f, indent=4)
    else:
        logger.warning("No tasks were successfully evaluated")


if __name__ == "__main__":
    main()