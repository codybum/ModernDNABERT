"""
Modified version of training/accelerate_utils.py that supports loading
pre-trained tokenizers instead of training them.
"""

import os
import math
import logging
import gc
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import torch
from torch.utils.data import DataLoader
from transformers import get_scheduler, BertForMaskedLM
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from tqdm import tqdm

from data.data_collator import GenomicDataset, GenomicMLMDataCollator
from modeling.alibi_attention import create_genomic_bert_config, create_genomic_bert_model
from tokenization.genomic_tokenizer import GenomicTokenizer
from training.train_utils import (
    generate_test_sequences,
    test_sequence_length_extrapolation,
    test_tokenizer_oov_handling,
    setup_mlm_data_collator
)

logger = logging.getLogger(__name__)


def ensure_valid_batch_indices(batch, tokenizer, model):
    """
    Ensure all token IDs in a batch are within the valid vocabulary range.

    Args:
        batch: The batch of inputs
        tokenizer: The tokenizer
        model: The model

    Returns:
        Updated batch with validated indices
    """
    # Original function code unchanged
    if 'input_ids' not in batch:
        return batch

    # Get model's vocabulary size (embedding dimension)
    embedding_size = model.get_input_embeddings().weight.shape[0]

    # Check if any indices are out of bounds
    input_ids = batch['input_ids']
    max_id = input_ids.max().item()

    if max_id >= embedding_size:
        logger.warning(f"Found token ID {max_id} exceeding model vocab size {embedding_size}")

        # Create a mask of out-of-bounds IDs
        invalid_mask = input_ids >= embedding_size

        if invalid_mask.any():
            # Count invalid indices
            num_invalid = invalid_mask.sum().item()
            logger.warning(f"Clamping {num_invalid} token IDs to valid range")

            # Replace out-of-bounds IDs with unk_token_id
            unk_token_id = getattr(tokenizer, 'unk_token_id', 0) if tokenizer else 0
            input_ids = torch.where(invalid_mask,
                                    torch.tensor(unk_token_id, device=input_ids.device, dtype=input_ids.dtype),
                                    input_ids)

            # Update the batch
            batch['input_ids'] = input_ids

            # If labels exist, update them too to avoid loss on invalid indices
            if 'labels' in batch:
                batch['labels'] = torch.where(invalid_mask,
                                              torch.tensor(-100, device=batch['labels'].device,
                                                           dtype=batch['labels'].dtype),
                                              batch['labels'])

    return batch


def modify_bert_attention(model, attention_type):
    """
    Modify a BERT model to use the specified attention type.

    Args:
        model: A BertForMaskedLM model instance
        attention_type: Type of attention mechanism to use ("standard" or "alibi")

    Returns:
        The modified model with the specified attention type
    """
    from modeling.alibi_attention import modify_bert_for_alibi

    # Original function code unchanged
    # Get current attention type (if set)
    current_attention_type = getattr(model.config, 'attention_type', 'standard')

    # If attention type is already set correctly, no need to modify
    if current_attention_type == attention_type:
        logger.info(f"Model already uses {attention_type} attention")
        return model

    if attention_type == "alibi":
        # Modify model to use ALiBi attention
        logger.info("Modifying model to use ALiBi attention")
        return modify_bert_for_alibi(model)
    elif attention_type == "standard":
        # If model is currently using ALiBi, revert to standard attention
        if getattr(model.config, 'use_alibi', False) or current_attention_type == 'alibi':
            logger.info("Reverting model to use standard attention")
            # We need to replace ALiBi attention with standard attention
            from transformers.models.bert.modeling_bert import BertSelfAttention

            # Reset any ALiBi-specific configurations
            model.config.use_alibi = False
            model.config.position_embedding_type = "absolute"
            model.config.attention_type = "standard"

            # Replace ALiBi attention with standard BertSelfAttention in each layer
            for i, layer in enumerate(model.bert.encoder.layer):
                old_attention = layer.attention.self

                # Create new standard attention
                new_attention = BertSelfAttention(model.config)

                # Copy weights from old attention to new attention
                new_attention.query = old_attention.query
                new_attention.key = old_attention.key
                new_attention.value = old_attention.value
                new_attention.dropout = old_attention.dropout

                # Replace the attention layer
                layer.attention.self = new_attention

                logger.info(f"Replaced ALiBi attention in layer {i} with standard attention")

            # Resize token embeddings to ensure compatibility
            model.resize_token_embeddings(len(model.get_input_embeddings().weight))

            logger.info("All layers successfully modified to use standard attention")
        else:
            logger.info("Model already uses standard attention")

        return model
    else:
        raise ValueError(f"Unsupported attention type: {attention_type}")


def train_with_accelerate(args, accelerator):
    """
    Main training function using Accelerate with support for selectable attention mechanisms.

    Modified to load pre-trained tokenizer instead of training it.

    Args:
        args: Command-line arguments
        accelerator: Accelerator instance
    """
    logger = logging.getLogger(__name__)

    # Set random seed for reproducibility
    if args.seed is not None:
        from accelerate.utils import set_seed
        set_seed(args.seed)

    # Create output directory
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    # Wait for all processes to sync
    accelerator.wait_for_everyone()

    # Load pre-trained tokenizer
    logger.info(f"Loading pre-trained tokenizer from {args.tokenizer_path}")
    try:
        tokenizer = GenomicTokenizer.from_pretrained(args.tokenizer_path)

        # Ensure special tokens are properly defined
        def _ensure_special_tokens(tokenizer):
            # Get vocabulary size for validation
            vocab_size = tokenizer.vocab_size

            # Fix mask_token_id directly if it exceeds vocab size
            if hasattr(tokenizer, 'mask_token_id') and tokenizer.mask_token_id is not None:
                if tokenizer.mask_token_id >= vocab_size:
                    # Use a fixed safe ID
                    safe_id = 1  # Use a simple, safe ID within vocab range

                    # Store the original token text
                    mask_token_text = tokenizer.mask_token

                    # Update the tokenizer's internal mappings
                    tokenizer.mask_token_id = safe_id

                    # Update the special tokens dictionary if it exists
                    if hasattr(tokenizer, 'special_tokens_map'):
                        tokenizer.special_tokens_map['mask_token'] = mask_token_text

                    # Make sure the vocab knows about this mapping
                    if hasattr(tokenizer, 'added_tokens_encoder'):
                        tokenizer.added_tokens_encoder[mask_token_text] = safe_id
                    if hasattr(tokenizer, 'added_tokens_decoder'):
                        tokenizer.added_tokens_decoder[safe_id] = mask_token_text

                    logger.info(f"Fixed mask_token to use ID {safe_id} instead of {tokenizer.mask_token_id}")

        _ensure_special_tokens(tokenizer)

        # Test tokenizer OOV handling
        if accelerator.is_main_process:
            try:
                test_tokenizer_oov_handling(tokenizer)
            except Exception as e:
                logger.error(f"Tokenizer test failed: {e}")
                logger.error("This indicates potential problems with token ID handling")

    except Exception as e:
        raise ValueError(f"Failed to load tokenizer from {args.tokenizer_path}: {e}")

    # Setup model with the specified attention type
    if args.model_path and os.path.exists(args.model_path):
        logger.info(f"Loading existing model from {args.model_path}")
        model = BertForMaskedLM.from_pretrained(args.model_path)

        # Check if model has correct attention type
        model_attention_type = getattr(model.config, 'attention_type', 'standard')
        if model_attention_type != args.attention_type:
            logger.info(f"Changing attention type from {model_attention_type} to {args.attention_type}")
            model = modify_bert_attention(model, args.attention_type)
    else:
        logger.info(f"Initializing new BERT model with {args.attention_type} attention for genomic data")
        config = create_genomic_bert_config(
            vocab_size=tokenizer.vocab_size,
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_hidden_layers,
            num_attention_heads=args.num_attention_heads,
            intermediate_size=args.hidden_size * 4,
            max_position_embeddings=args.pre_training_length,
            use_alibi=args.attention_type == "alibi",  # Map attention_type to use_alibi boolean
            attention_type=args.attention_type,  # Pass the attention_type string as well
            max_supported_length=args.max_supported_model_length,
        )
        model = create_genomic_bert_model(config)

    # CRITICAL FIX: Verify and synchronize model-tokenizer compatibility
    model = verify_model_tokenizer_compatibility(model, tokenizer)

    # Prepare dataset
    train_dataset = GenomicDataset(
        args.input_files,
        tokenizer,
        pre_training_length=args.pre_training_length,
        max_inference_length=args.max_inference_length,
        mlm_probability=args.mlm_probability,
        chunk_size=args.chunk_size,
        stride=args.stride,
        sample_long_sequences=args.sample_long_sequences,
        max_safe_sequence_length=args.max_safe_sequence_length,
    )

    data_collator = setup_mlm_data_collator(
        tokenizer=tokenizer,
        mlm_probability=args.mlm_probability,
        model=model,
        max_seq_length=args.pre_training_length  # Pass pre_training_length to collator
    )

    # Configure data loader
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        collate_fn=data_collator,
        shuffle=True,
        num_workers=0,  # Set to 0 to avoid multiprocessing pickling issues
        pin_memory=torch.cuda.is_available(),
    )

    # Prepare optimizer
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Calculate training steps
    num_update_steps_per_epoch, max_train_steps = calculate_training_steps(
        train_dataloader, args.gradient_accumulation_steps, args.epochs
    )

    # Setup learning rate scheduler
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=max_train_steps,
    )

    # Apply sequence length overrides for testing if specified
    if args.test_sequence_length is not None:
        logger.info(f"TEST MODE: Enforcing sequence length of {args.test_sequence_length}")
        args.pre_training_length = args.test_sequence_length
        args.max_supported_model_length = args.test_sequence_length
        model.config.max_supported_length = min(getattr(model.config, "max_supported_length", 4096), 4096)

    # Prepare with accelerator
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    # Generate test sequences for extrapolation testing
    test_lengths = [
        args.pre_training_length,
        args.pre_training_length * 2,
        args.pre_training_length * 4,
    ]
    if args.max_inference_length:
        test_lengths.append(args.max_inference_length)

    extrapolation_test_seqs = generate_test_sequences(test_lengths)

    # Test initial extrapolation
    if accelerator.is_main_process:
        logger.info("\nInitial length extrapolation test:")
        test_sequence_length_extrapolation(accelerator, model, tokenizer, extrapolation_test_seqs)

    # Resume from checkpoint if needed
    starting_epoch, completed_steps = resume_from_checkpoint(accelerator, args, num_update_steps_per_epoch)

    # Compute total training batch size
    total_batch_size = args.batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    # Log training information
    logger.info("***** Running training *****")
    logger.info(f"  Attention type = {args.attention_type}")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num epochs = {args.epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")

    # Setup progress bar (only on main process)
    progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_main_process)
    progress_bar.update(completed_steps)

    # Main training loop
    for epoch in range(starting_epoch, args.epochs):
        model.train()
        total_loss = 0
        valid_steps = 0  # Count successful steps for averaging

        # Report GPU memory status on main process
        if torch.cuda.is_available() and accelerator.is_main_process:
            logger.info(f"Starting epoch {epoch} - GPU memory status:")
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / (1024 * 1024)
                reserved = torch.cuda.memory_reserved(i) / (1024 * 1024)
                logger.info(f"  GPU {i}: Allocated: {allocated:.2f} MB, Reserved: {reserved:.2f} MB")

        # Process batches with improved error handling
        for step, batch in enumerate(train_dataloader):
            # Process batch safely
            loss = safe_training_step(model, batch, accelerator, args)

            if loss is not None:
                # Successfully processed batch
                total_loss += loss
                valid_steps += 1

                # Update on accumulation steps
                if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                    # Clip gradients
                    if args.max_grad_norm is not None:
                        accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                    # Optimizer and scheduler step
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                    # Update progress
                    progress_bar.update(1)
                    completed_steps += 1

                    # Log loss periodically
                    if completed_steps % args.logging_steps == 0 and valid_steps > 0:
                        avg_loss = total_loss.item() / valid_steps
                        logger.info(f"Epoch: {epoch}, Step: {completed_steps}, Loss: {avg_loss:.4f}")
                        total_loss = 0
                        valid_steps = 0

                    # Save checkpoint if needed
                    if args.checkpointing_steps is not None and completed_steps % args.checkpointing_steps == 0:
                        save_checkpoint(
                            accelerator, args, epoch, completed_steps,
                            model, optimizer, lr_scheduler, tokenizer
                        )
            else:
                # Error in this batch - skip and continue
                logger.warning(f"Skipping problematic batch at Epoch {epoch}, Step {step}")
                # Explicit synchronization
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()

        # Save checkpoint after each epoch
        save_checkpoint(
            accelerator, args, epoch, completed_steps,
            model, optimizer, lr_scheduler, tokenizer
        )

    # Save final model
    if accelerator.is_main_process:
        final_dir = os.path.join(args.output_dir, "final")
        save_model(accelerator, model, tokenizer, final_dir)

    # Final length extrapolation test (only on main process)
    if accelerator.is_main_process:
        logger.info("\nFinal length extrapolation test:")
        test_sequence_length_extrapolation(accelerator, model, tokenizer, extrapolation_test_seqs)

    logger.info("Training complete!")


# The rest of the helper functions remain unchanged from the original file
def setup_accelerator(args):
    # Create checkpoint configuration
    project_config = None
    if args.output_dir:
        project_config = ProjectConfiguration(
            project_dir=args.output_dir,
            logging_dir=os.path.join(args.output_dir, "logs")
        )

    # Set up device configuration with explicit device IDs
    device_placement_config = None
    if torch.cuda.is_available():
        # Explicitly map processes to devices
        device_placement_config = {"device_map": "auto"}

    # Create accelerator with explicit device mapping
    accelerator_config = {
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "project_config": project_config,
        "device_placement": device_placement_config,
    }

    if hasattr(args, 'log_with_tensorboard') and args.log_with_tensorboard:
        accelerator_config["log_with"] = "tensorboard"

    # Create accelerator
    accelerator = Accelerator(**accelerator_config)

    return accelerator


def calculate_training_steps(train_dataloader, gradient_accumulation_steps, num_epochs):
    """Calculate the number of update steps for training."""
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
    max_train_steps = num_epochs * num_update_steps_per_epoch
    return num_update_steps_per_epoch, max_train_steps


def save_checkpoint(accelerator, args, epoch, step, model, optimizer, lr_scheduler, tokenizer):
    """Save a checkpoint with proper state management."""
    # Determine checkpoint path (epoch or step-based)
    if step % args.save_steps == 0:
        checkpoint_path = os.path.join(args.output_dir, f"step_{step}")
    else:
        checkpoint_path = os.path.join(args.output_dir, f"epoch_{epoch}")

    logger.info(f"Saving checkpoint to {checkpoint_path}")

    # Use Accelerate's state saving
    accelerator.save_state(checkpoint_path)

    # Save tokenizer on main process only
    if accelerator.is_main_process and tokenizer is not None:
        tokenizer.save_pretrained(os.path.join(checkpoint_path, "tokenizer"))

    # Clean up old checkpoints (only keep N most recent)
    if accelerator.is_main_process and args.save_total_limit is not None:
        cleanup_checkpoints(args.output_dir, args.save_total_limit)

    # Wait for checkpoint saving to complete
    accelerator.wait_for_everyone()


def cleanup_checkpoints(output_dir, save_total_limit):
    """Clean up old checkpoints, keeping only the most recent ones."""
    import os
    import shutil
    import re

    # Get all checkpoint directories
    checkpoint_dirs = []
    pattern = re.compile(r"(epoch|step)_(\d+)")

    for dirname in os.listdir(output_dir):
        if not os.path.isdir(os.path.join(output_dir, dirname)):
            continue

        match = pattern.match(dirname)
        if match:
            checkpoint_type = match.group(1)  # 'epoch' or 'step'
            checkpoint_num = int(match.group(2))
            checkpoint_dirs.append((dirname, checkpoint_type, checkpoint_num))

    # Sort by type and number
    checkpoint_dirs.sort(key=lambda x: (x[1], x[2]))

    # Keep only the most recent ones
    if len(checkpoint_dirs) > save_total_limit:
        dirs_to_remove = checkpoint_dirs[:-save_total_limit]
        for dirname, _, _ in dirs_to_remove:
            logger.info(f"Removing old checkpoint: {dirname}")
            shutil.rmtree(os.path.join(output_dir, dirname))


def resume_from_checkpoint(accelerator, args, num_update_steps_per_epoch):
    """Resume training from a checkpoint with proper state tracking."""
    starting_epoch = 0
    completed_steps = 0

    if not args.resume_from_checkpoint:
        return starting_epoch, completed_steps

    checkpoint_path = args.resume_from_checkpoint

    # Find latest checkpoint if requested
    if args.resume_from_checkpoint == "latest":
        # Look in both epoch-based and step-based directories
        checkpoint_dirs = []

        # Check epoch directories
        epoch_dirs = [d for d in os.listdir(args.output_dir)
                      if d.startswith("epoch_") and os.path.isdir(os.path.join(args.output_dir, d))]
        checkpoint_dirs.extend([(d, int(d.split("_")[-1])) for d in epoch_dirs])

        # Check step directories
        step_dirs = [d for d in os.listdir(args.output_dir)
                     if d.startswith("step_") and os.path.isdir(os.path.join(args.output_dir, d))]
        checkpoint_dirs.extend([(d, int(d.split("_")[-1])) for d in step_dirs])

        if not checkpoint_dirs:
            logger.info("No checkpoints found, starting training from scratch")
            return starting_epoch, completed_steps

        # Sort by number (either epoch or step)
        checkpoint_dirs.sort(key=lambda x: x[1])
        latest_dir = checkpoint_dirs[-1][0]
        checkpoint_path = os.path.join(args.output_dir, latest_dir)

    # Log checkpoint info
    logger.info(f"Resuming from checkpoint: {checkpoint_path}")

    # Load checkpoint state
    accelerator.load_state(checkpoint_path)

    # Extract epoch and step from checkpoint path
    path = Path(checkpoint_path)
    if "epoch_" in path.name:
        starting_epoch = int(path.name.split("_")[-1])
        completed_steps = starting_epoch * num_update_steps_per_epoch
    elif "step_" in path.name:
        completed_steps = int(path.name.split("_")[-1])
        starting_epoch = completed_steps // num_update_steps_per_epoch

    logger.info(f"Resuming from epoch {starting_epoch}, step {completed_steps}")

    # Force synchronize CUDA to ensure clean resumption
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    return starting_epoch, completed_steps


def save_model(accelerator, model, tokenizer, output_dir):
    """Save the model and tokenizer using Accelerate."""
    # Create output directory if needed
    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)

    # Wait for directory creation
    accelerator.wait_for_everyone()

    # Unwrap model and save on main process
    unwrapped_model = accelerator.unwrap_model(model)

    if accelerator.is_main_process:
        # Temporarily remove tokenizer from config if present
        tokenizer_from_config = None
        if hasattr(unwrapped_model.config, 'tokenizer'):
            tokenizer_from_config = unwrapped_model.config.tokenizer
            delattr(unwrapped_model.config, 'tokenizer')

        try:
            # Save model and config using HuggingFace's built-in methods
            unwrapped_model.save_pretrained(output_dir)
            logger.info(f"Model saved to {output_dir}")
        finally:
            # Restore tokenizer in config
            if tokenizer_from_config is not None:
                unwrapped_model.config.tokenizer = tokenizer_from_config

        # Save tokenizer if provided
        if tokenizer is not None:
            tokenizer_path = os.path.join(output_dir, "tokenizer")
            tokenizer.save_pretrained(tokenizer_path)
            logger.info(f"Tokenizer saved to {tokenizer_path}")

    # Wait for saving to complete
    accelerator.wait_for_everyone()


def verify_model_tokenizer_compatibility(model, tokenizer):
    # First, fix any token ID issues in the tokenizer
    if hasattr(tokenizer, 'mask_token_id') and tokenizer.mask_token_id is not None:
        if tokenizer.mask_token_id >= tokenizer.vocab_size:
            tokenizer.mask_token_id = 1  # Use consistent safe ID
            logger.warning(f"Fixed tokenizer mask_token_id to use ID 1")

    # Then, add this to model config
    model.config.mask_token_id = tokenizer.mask_token_id

    # Ensure embedding size matches
    new_size = tokenizer.vocab_size
    model.resize_token_embeddings(new_size)

    # Verify the change was successful
    if model.get_input_embeddings().weight.shape[0] != new_size:
        logger.error(f"Failed to resize model embeddings to {new_size}")

    return model


def safe_training_step(model, batch, accelerator, args):
    """Execute a single training step with robust error handling."""
    try:
        # Validate token IDs before forward pass
        tokenizer = getattr(accelerator.unwrap_model(model).config, 'tokenizer', None)
        batch = ensure_valid_batch_indices(batch, tokenizer, model)

        # Check available GPU memory
        if torch.cuda.is_available():
            device = next(model.parameters()).device
            available_memory = torch.cuda.get_device_properties(device).total_memory - torch.cuda.memory_allocated(
                device)
            available_gb = available_memory / (1024 ** 3)
            logger.debug(f"Available GPU memory: {available_gb:.2f} GB")

        # Get batch dimensions
        if 'input_ids' in batch:
            batch_size, seq_length = batch['input_ids'].shape

            # Implement dynamic handling based on sequence length
            # Longer sequences need more memory - adjust batch dynamically if needed
            if seq_length > 2048:
                # For very long sequences, split batch if memory is tight
                if torch.cuda.is_available() and available_gb < 2.0:  # Conservative threshold
                    logger.warning(
                        f"Sequence length {seq_length} with low memory ({available_gb:.2f} GB). Processing in micro-batches.")
                    return _process_in_micro_batches(model, batch, accelerator, args)

        # Forward pass
        outputs = model(**batch)
        loss = outputs.loss / args.gradient_accumulation_steps

        # Backward pass with gradient scaling if FP16 is enabled
        accelerator.backward(loss)

        return loss.detach().float()

    except RuntimeError as e:
        error_msg = str(e)

        # Handle different error types
        if "CUDA out of memory" in error_msg:
            logger.error(f"CUDA OOM error during training: {e}")

            # Try to recover by processing in micro-batches
            if 'input_ids' in batch and batch['input_ids'].shape[0] > 1:
                logger.info("Attempting recovery by processing in micro-batches")
                try:
                    return _process_in_micro_batches(model, batch, accelerator, args)
                except Exception as micro_error:
                    logger.error(f"Micro-batch recovery failed: {micro_error}")

            # Free memory
            torch.cuda.empty_cache()
            gc.collect()

        elif "index out of bounds" in error_msg or "indexSelectLargeIndex" in error_msg:
            logger.error(f"CUDA indexing error: {e}")
            # Log input shapes to help debug
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    logger.error(f"  {k} shape: {v.shape}, min: {v.min().item()}, max: {v.max().item()}")
        else:
            logger.error(f"Unexpected error: {e}")

        # Force synchronize
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        return None


def _process_in_micro_batches(model, batch, accelerator, args, micro_batch_size=1):
    """Process a batch in smaller micro-batches to handle memory constraints."""
    # Get the original batch size
    full_batch_size = batch['input_ids'].shape[0]

    # If batch size is already 1, we can't split further
    if full_batch_size <= micro_batch_size:
        raise ValueError("Batch size is already at minimum, cannot process in micro-batches")

    # Initialize accumulated loss using accelerator's device
    accumulated_loss = torch.tensor(0.0, device=accelerator.device)

    # Process each micro-batch
    for i in range(0, full_batch_size, micro_batch_size):
        # Get micro-batch indices
        end_idx = min(i + micro_batch_size, full_batch_size)
        logger.info(
            f"Processing micro-batch {i // micro_batch_size + 1}/{(full_batch_size + micro_batch_size - 1) // micro_batch_size}")

        # Create micro-batch
        micro_batch = {k: v[i:end_idx] for k, v in batch.items()}

        # Validate token IDs in micro-batch
        tokenizer = getattr(accelerator.unwrap_model(model).config, 'tokenizer', None)
        micro_batch = ensure_valid_batch_indices(micro_batch, tokenizer, model)

        # Forward pass
        outputs = model(**micro_batch)
        micro_loss = outputs.loss / args.gradient_accumulation_steps / (full_batch_size / micro_batch_size)

        # Backward pass - use accelerator
        accelerator.backward(micro_loss)

        # Add to accumulated loss
        accumulated_loss += micro_loss.detach() * (end_idx - i)

    # Return average loss
    return accumulated_loss / full_batch_size